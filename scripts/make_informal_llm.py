"""Generate the LLM-paraphrase informal variant of a query set (Phase 2).

The `llm` arm of Phase 2: rewrite formal MIRACL-id queries into heavy colloquial
Indonesian with an LLM, few-shot-primed in STIF / Kamus-Alay register. Meaning is
preserved so the formal qrels stay valid (verify later with
src/eval/semantic_preservation.py before trusting the set).

Model host: Azure AI Foundry / Azure OpenAI, deployment = gpt-5-mini. Configure
via environment (a .env in repo root is auto-loaded if python-dotenv is present):

    AZURE_OPENAI_ENDPOINT     https://<resource>.openai.azure.com/  (or Foundry endpoint)
    AZURE_OPENAI_API_KEY      <key>
    AZURE_OPENAI_DEPLOYMENT   gpt-5-mini            (your deployment name)
    AZURE_OPENAI_API_VERSION  2024-12-01-preview    (optional; default below)

    pip install openai            # not in requirements.txt (inference-only, no GPU)

Usage:
    python scripts/make_informal_llm.py --dry-run          # print the prompt, no API call
    python scripts/make_informal_llm.py --limit 10         # smoke test 10 queries
    python scripts/make_informal_llm.py --workers 8        # full pilot set
    python scripts/run_gap_eval.py --tag gap-llm --variants experiments/pilot/pilot_informal_llm.tsv

The output TSV (query_id, `llm`) is written incrementally and is resumable:
re-running skips query ids already present, so an interrupted or rate-limited run
just continues. Rows that error out are left missing and retried on the next run.
"""
import os
import sys

if sys.platform == "win32" and sys.flags.utf8_mode == 0:
    import subprocess

    sys.exit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))

import argparse
import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_API_VERSION = "2024-12-01-preview"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env(path: str = os.path.join(_REPO_ROOT, ".env")) -> None:
    """Load KEY=VALUE lines from repo-root .env into os.environ (no overwrite).

    Uses python-dotenv if present, else a minimal built-in parser so the script
    has no extra dependency. Existing environment variables always win.
    """
    if not os.path.exists(path):
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(path, override=False)
        return
    except ImportError:
        pass
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip("'\""))

SYSTEM_PROMPT = """\
Kamu penutur asli bahasa Indonesia yang mengubah pertanyaan formal menjadi \
gaya informal/gaul seperti orang chat sehari-hari (WhatsApp, X, forum).

Aturan:
- PERTAHANKAN makna dan kebutuhan informasi persis sama. Jangan menambah atau \
menghapus entitas, nama, angka, atau maksud pertanyaan.
- Ubah HANYA register: pakai slang, singkatan, klitik (gue/lo/gw), partikel \
(sih, dong, deh, nih, kan), ejaan santai (yg, gak, udah, kalo, bgt), dan gaya \
ngobrol. Boleh drop tanda baca/kapital.
- Tetap satu pertanyaan yang wajar diketik orang. Jangan kasih penjelasan.
- Balas HANYA JSON: {"informal": "<hasil rewrite>"}"""

# Few-shot pairs (formal -> heavy informal), primed on STIF / Kamus-Alay style.
FEWSHOT = [
    ("Apa yang dimaksud dengan artefak?", "artefak tu maksudnya apaan sih? ada yg tau ga"),
    ("Di kendaraan apakah dapat ditemukan mesin pembakaran dalam?",
     "mesin pembakaran dalam tuh biasa ada di kendaraan apa ya?"),
    ("Pada umur berapa Paul David Hewson bergabung dengan U2?",
     "si Paul David Hewson gabung U2 pas umur berapa sih"),
    ("Apa penyebab menstruasi?", "menstruasi tuh penyebabnya apaan sih sebenernya"),
]


def build_messages(formal: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for f, i in FEWSHOT:
        messages.append({"role": "user", "content": f})
        messages.append({"role": "assistant", "content": json.dumps({"informal": i}, ensure_ascii=False)})
    messages.append({"role": "user", "content": formal})
    return messages


def load_formal(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows or "formal" not in rows[0]:
        sys.exit(f"[llm] {path} has no 'formal' column")
    return {r["query_id"]: r["formal"].strip() for r in rows if r["formal"].strip()}


def load_done(path: str) -> set[str]:
    """Query ids already written with a non-empty rewrite (for --resume)."""
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8", newline="") as f:
        return {r["query_id"] for r in csv.DictReader(f, delimiter="\t") if r.get("llm", "").strip()}


def make_client(api_version: str):
    try:
        from openai import AzureOpenAI
    except ImportError:
        sys.exit("[llm] openai not installed — run: pip install openai")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not endpoint or not key:
        sys.exit("[llm] set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY (see module docstring)")
    return AzureOpenAI(azure_endpoint=endpoint, api_key=key, api_version=api_version)


def rewrite(client, deployment: str, formal: str, reasoning: str, max_tokens: int, retries: int = 4) -> str:
    """One rewrite with retry/backoff; returns the informal string (JSON-parsed)."""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=deployment,
                messages=build_messages(formal),
                response_format={"type": "json_object"},
                max_completion_tokens=max_tokens,
                extra_body={"reasoning_effort": reasoning},
            )
            content = resp.choices[0].message.content or ""
            try:
                out = json.loads(content).get("informal", "").strip()
            except json.JSONDecodeError:
                out = content.strip()
            if out:
                return out
            raise ValueError("empty rewrite")
        except Exception as e:  # noqa: BLE001 — transient API/rate errors, back off and retry
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def main() -> None:
    load_env()  # before argparse so .env can supply default deployment / api-version

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--queries", default="experiments/pilot/pilot_queries.tsv",
                    help="source TSV with a 'formal' column (ignored if --from-miracl)")
    ap.add_argument("--from-miracl", action="store_true",
                    help="rewrite the full MIRACL-id dev split instead of the pilot")
    ap.add_argument("--out", default="experiments/pilot/pilot_informal_llm.tsv")
    ap.add_argument("--deployment", default=None, help="override AZURE_OPENAI_DEPLOYMENT")
    ap.add_argument("--api-version", default=os.environ.get("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION))
    ap.add_argument("--reasoning-effort", default="low", choices=["minimal", "low", "medium", "high"])
    ap.add_argument("--max-tokens", type=int, default=512, help="max_completion_tokens per call")
    ap.add_argument("--workers", type=int, default=4, help="concurrent requests")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N pending queries")
    ap.add_argument("--no-resume", action="store_true", help="ignore existing output and redo all")
    ap.add_argument("--dry-run", action="store_true", help="print the prompt for one query and exit")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.from_miracl:
        from src.data import miracl
        formal = miracl.load_queries(cfg["data"]["split"])
    else:
        formal = load_formal(args.queries)

    if args.dry_run:
        qid, text = next(iter(formal.items()))
        print(f"[llm] dry-run prompt for query {qid}:\n")
        for m in build_messages(text):
            print(f"--- {m['role']} ---\n{m['content']}\n")
        print("[llm] no API call made.")
        return

    deployment = args.deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    if not deployment:
        sys.exit("[llm] set AZURE_OPENAI_DEPLOYMENT or pass --deployment (e.g. gpt-5-mini)")

    done = set() if args.no_resume else load_done(args.out)
    pending = [(q, t) for q, t in formal.items() if q not in done]
    if args.limit is not None:
        pending = pending[: args.limit]
    if not pending:
        print(f"[llm] nothing to do — {len(done)} already in {args.out}")
        return

    client = make_client(args.api_version)
    write_header = args.no_resume or not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    mode = "w" if args.no_resume else "a"
    lock = threading.Lock()
    ok = fail = 0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, mode, encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        if write_header:
            writer.writerow(["query_id", "llm"])
            fh.flush()

        def work(item):
            qid, text = item
            return qid, rewrite(client, deployment, text, args.reasoning_effort, args.max_tokens)

        print(f"[llm] {deployment} via {args.api_version}, reasoning={args.reasoning_effort}, "
              f"{len(pending)} queries, {args.workers} workers")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(work, item): item[0] for item in pending}
            for fut in as_completed(futures):
                qid = futures[fut]
                try:
                    qid, informal = fut.result()
                except Exception as e:  # noqa: BLE001
                    fail += 1
                    print(f"[llm]   FAIL {qid}: {e}")
                    continue
                with lock:
                    writer.writerow([qid, informal])
                    fh.flush()
                    ok += 1
                    if ok % 25 == 0:
                        print(f"[llm]   {ok} done...")

    print(f"[llm] wrote {ok} rewrites to {args.out} ({fail} failed, {len(done)} pre-existing)")
    if fail:
        print("[llm] re-run to retry failed queries (resume skips the ones that succeeded)")
    else:
        print(f"[llm] feed to: python scripts/run_gap_eval.py --tag gap-llm --variants {args.out}")


if __name__ == "__main__":
    main()
