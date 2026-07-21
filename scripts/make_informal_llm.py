"""Generate the LLM-paraphrase informal variant of a query set (Phase 2).

The `llm` arm of Phase 2: rewrite formal MIRACL-id queries into heavy Twitter/X
register Indonesian with an LLM, few-shot-primed for aggressive slang. Named
entities, numbers, and key terms are kept spelled out in full (only register,
spelling, and structure are informalized) so the formal qrels stay valid — this
is the fix for the "Maluku Utara -> malut" entity-mangling seen in the manual
heavy set. Verify with src/eval/semantic_preservation.py before trusting the set.

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
import random
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
Kamu netizen Indonesia. Tulis ULANG pertanyaan formal jadi cuitan gaya Twitter/X \
yang santai banget: huruf kecil semua, tanpa tanda baca, banyak singkatan dan \
partikel, kayak orang nanya sambil ngobrol di timeline.

WAJIB gaya berat (heavy informal):
- huruf kecil semua, buang tanda baca dan kapital.
- pakai klitik & slang: gue/lo/gw, gaes, cuy, gan, dong, sih, deh, nih, kan, anjir, ya.
- singkat kata umum: yang->yg, tidak->gak/ga, sudah->udah, dengan->sama, \
kenapa->knp, berapa->brp, tahun->taun, orang->org.
- ubah susunan kalimat jadi gaya ngobrol (topik dulu baru nanya).

VARIASI (penting):
- Tiap pertanyaan gayanya beda-beda. JANGAN pakai frasa penutup yang sama \
berulang. Hindari selalu menutup dengan "ada yg tau ga", "jelasin dong", \
"kasih tau dong" — itu terlalu monoton.
- Sering-sering tanya LANGSUNG tanpa embel-embel penutup. Kalau mau kasih \
partikel, ganti-ganti (sih, ya, dah, deh, kan, kok, anjir, woy) dan taruh di \
tempat berbeda, bukan template yang sama.

DILARANG KERAS (bikin makna rusak):
- JANGAN singkat, terjemahkan, atau ganti NAMA DIRI / ENTITAS / ISTILAH KUNCI. \
Nama orang, tempat, organisasi, judul, dan istilah teknis harus DITULIS UTUH \
(boleh huruf kecil, tapi kata-katanya lengkap).
  SALAH: "Maluku Utara" -> "malut"   |   BENAR: "Maluku Utara" -> "maluku utara"
  SALAH: "Perang Dunia II" -> "pd2"   |   BENAR: -> "perang dunia II"
- JANGAN ubah angka, tanggal, atau satuan. JANGAN menambah/menghapus maksud.

Balas HANYA JSON: {"informal": "<hasil rewrite>"}"""

# Few-shot POOL (formal -> heavy Twitter/X register). Deliberately varied endings
# so the model doesn't template one closer; build_messages samples a rotating
# subset per query. Each keeps named entities, numbers, and key terms spelled out
# in full — only register/spelling/structure is informalized. The Maluku Utara
# pair is the anchor against entity mangling.
FEWSHOT_POOL = [
    ("Apa yang dimaksud dengan artefak?", "artefak tu maksudnya apaan sih"),
    ("Berapa luas Maluku Utara?", "eh luas Maluku Utara brp ya"),
    ("Pada umur berapa Paul David Hewson bergabung dengan U2?",
     "Paul David Hewson gabung U2 pas umur brp dah penasaran"),
    ("Berapa populasi pemeluk agama Islam di Indonesia pada tahun 2010?",
     "org islam di Indonesia taun 2010 brp banyak sih jumlahnya"),
    ("Siapa penemu bola lampu?", "yg nemuin bola lampu siapa dah"),
    ("Apa penyebab menstruasi?", "menstruasi tu kenapa bisa terjadi kok"),
    ("Kapan Perang Dunia II berakhir?", "Perang Dunia II kelar taun brp"),
    ("Di mana letak Gunung Semeru?", "Gunung Semeru tuh letaknya dimana ya woy"),
]
N_SHOTS = 5  # examples shown per call, sampled per query for variety

# --lean: same heavy register, but NO padding — forbids adding clauses about the
# asker's state ("gue penasaran / blom hapal / mau masukin laporan") that carry no
# information but drag the query embedding off the entity. Isolates the register
# effect from padding noise (see the semantic-preservation finding).
LEAN_RULE = """

HEMAT KATA (mode lean):
- JANGAN tambah klausa/curhat soal keadaan penanya: "gue penasaran", "blom hapal", \
"masih bingung", "lagi kepo", "mau masukin laporan", "baru denger" — DILARANG.
- Cukup pertanyaannya saja + maksimal SATU partikel pendek (sih/ya/dong). Jangan \
lebih panjang dari pertanyaan aslinya."""

FEWSHOT_POOL_LEAN = [
    ("Apa yang dimaksud dengan artefak?", "artefak tu maksudnya apaan sih"),
    ("Berapa luas Maluku Utara?", "luas Maluku Utara brp ya"),
    ("Pada umur berapa Paul David Hewson bergabung dengan U2?",
     "Paul David Hewson gabung U2 umur brp"),
    ("Berapa populasi pemeluk agama Islam di Indonesia pada tahun 2010?",
     "pemeluk islam di Indonesia taun 2010 brp sih"),
    ("Siapa penemu bola lampu?", "penemu bola lampu siapa sih"),
    ("Apa penyebab menstruasi?", "menstruasi penyebabnya apa"),
    ("Kapan Perang Dunia II berakhir?", "Perang Dunia II kelar taun brp"),
    ("Di mana letak Gunung Semeru?", "Gunung Semeru letaknya dimana ya"),
]


def build_messages(formal: str, lean: bool = False) -> list[dict]:
    # Rotate a per-query subset of the pool (seeded by the query, so reproducible)
    # so the few-shot priming isn't identical across calls and won't template one closer.
    pool = FEWSHOT_POOL_LEAN if lean else FEWSHOT_POOL
    system = SYSTEM_PROMPT + LEAN_RULE if lean else SYSTEM_PROMPT
    rng = random.Random(formal)
    shots = rng.sample(pool, k=min(N_SHOTS, len(pool)))
    messages = [{"role": "system", "content": system}]
    for f, i in shots:
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


def load_done(path: str, column: str) -> set[str]:
    """Query ids already written with a non-empty rewrite (for --resume)."""
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8", newline="") as f:
        return {r["query_id"] for r in csv.DictReader(f, delimiter="\t") if r.get(column, "").strip()}


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


def rewrite(client, deployment: str, formal: str, reasoning: str, max_tokens: int,
            lean: bool = False, retries: int = 4) -> str:
    """One rewrite with retry/backoff; returns the informal string (JSON-parsed)."""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=deployment,
                messages=build_messages(formal, lean),
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
    ap.add_argument("--lean", action="store_true",
                    help="heavy register WITHOUT filler padding (isolates register from padding noise)")
    ap.add_argument("--out", default=None, help="output TSV (default depends on --lean)")
    ap.add_argument("--column", default=None, help="variant column name (default: llm_lean if --lean else llm)")
    ap.add_argument("--deployment", default=None, help="override AZURE_OPENAI_DEPLOYMENT")
    ap.add_argument("--api-version", default=os.environ.get("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION))
    ap.add_argument("--reasoning-effort", default="low", choices=["minimal", "low", "medium", "high"])
    ap.add_argument("--max-tokens", type=int, default=512, help="max_completion_tokens per call")
    ap.add_argument("--workers", type=int, default=4, help="concurrent requests")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N pending queries")
    ap.add_argument("--no-resume", action="store_true", help="ignore existing output and redo all")
    ap.add_argument("--dry-run", action="store_true", help="print the prompt for one query and exit")
    args = ap.parse_args()

    column = args.column or ("llm_lean" if args.lean else "llm")
    out = args.out or (f"experiments/pilot/pilot_informal_{column}.tsv")

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.from_miracl:
        from src.data import miracl
        formal = miracl.load_queries(cfg["data"]["split"])
    else:
        formal = load_formal(args.queries)

    if args.dry_run:
        qid, text = next(iter(formal.items()))
        print(f"[llm] dry-run prompt for query {qid} (lean={args.lean}):\n")
        for m in build_messages(text, args.lean):
            print(f"--- {m['role']} ---\n{m['content']}\n")
        print("[llm] no API call made.")
        return

    deployment = args.deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    if not deployment:
        sys.exit("[llm] set AZURE_OPENAI_DEPLOYMENT or pass --deployment (e.g. gpt-5-mini)")

    done = set() if args.no_resume else load_done(out, column)
    pending = [(q, t) for q, t in formal.items() if q not in done]
    if args.limit is not None:
        pending = pending[: args.limit]
    if not pending:
        print(f"[llm] nothing to do — {len(done)} already in {out}")
        return

    client = make_client(args.api_version)
    write_header = args.no_resume or not os.path.exists(out) or os.path.getsize(out) == 0
    mode = "w" if args.no_resume else "a"
    lock = threading.Lock()
    ok = fail = 0

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, mode, encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        if write_header:
            writer.writerow(["query_id", column])
            fh.flush()

        def work(item):
            qid, text = item
            return qid, rewrite(client, deployment, text, args.reasoning_effort, args.max_tokens, args.lean)

        print(f"[llm] {deployment} via {args.api_version}, reasoning={args.reasoning_effort}, "
              f"lean={args.lean}, col={column}, {len(pending)} queries, {args.workers} workers")
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

    print(f"[llm] wrote {ok} rewrites to {out} ({fail} failed, {len(done)} pre-existing)")
    if fail:
        print("[llm] re-run to retry failed queries (resume skips the ones that succeeded)")
    else:
        print(f"[llm] feed to: python scripts/run_gap_eval.py --tag gap-{column} --variants {out}")


if __name__ == "__main__":
    main()
