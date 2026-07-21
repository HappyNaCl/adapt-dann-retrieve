"""Generate the lexicon-injection informal variant of a query set (Phase 2).

Reads formal queries and writes a variants TSV whose single variant column
(`lexicon`) is the deterministic lexicon-injected informal rewrite. The output
plugs straight into the Phase 1 gap harness:

    python scripts/make_informal_set.py --rate 0.8
    python scripts/run_gap_eval.py --tag gap-lexicon \
        --variants experiments/pilot/pilot_informal_lexicon.tsv

By default it reuses the `formal` column of the pilot set so the query ids line
up with what run_gap_eval evaluates; pass --from-miracl to inject the full dev
split instead. Edit resources/lexicons/injection_map.tsv and rerun to change the
substitutions.
"""
import os
import sys

# ir_datasets opens MIRACL TSVs with the locale codepage (cp1252 on Windows)
# unless Python runs in UTF-8 mode, crashing on Indonesian text.
if sys.platform == "win32" and sys.flags.utf8_mode == 0:
    import subprocess

    sys.exit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))

import argparse
import csv

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import make_informal


def load_formal(path: str) -> dict[str, str]:
    """Read {query_id: formal} from a pilot TSV's `formal` column."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows or "formal" not in rows[0]:
        sys.exit(f"[informal] {path} has no 'formal' column")
    return {r["query_id"]: r["formal"].strip() for r in rows if r["formal"].strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--queries", default="experiments/pilot/pilot_queries.tsv",
                    help="source TSV with a 'formal' column (ignored if --from-miracl)")
    ap.add_argument("--from-miracl", action="store_true",
                    help="inject the full MIRACL-id dev split instead of the pilot")
    ap.add_argument("--out", default="experiments/pilot/pilot_informal_lexicon.tsv")
    ap.add_argument("--rate", type=float, default=1.0,
                    help="per-match swap probability in [0,1] (injection intensity)")
    ap.add_argument("--first-only", action="store_true",
                    help="always use each entry's default alternative (ignores --rate)")
    ap.add_argument("--seed", type=int, default=None, help="override config seed")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = args.seed if args.seed is not None else cfg["seed"]

    if args.from_miracl:
        from src.data import miracl
        formal = miracl.load_queries(cfg["data"]["split"])
    else:
        formal = load_formal(args.queries)

    mapping = make_informal.load_injection_map()
    informal = make_informal.inject_queries(
        formal, mapping, seed=seed, rate=args.rate, first_only=args.first_only,
    )

    changed = sum(1 for q in formal if informal[q] != formal[q])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["query_id", "lexicon"])
        for qid in formal:
            writer.writerow([qid, informal[qid]])

    print(f"[informal] {len(mapping)} map entries, seed={seed}, rate={args.rate}"
          f"{' (first-only)' if args.first_only else ''}")
    print(f"[informal] wrote {len(formal)} queries to {args.out} "
          f"({changed} changed, {len(formal) - changed} untouched)")
    print("[informal] feed to: python scripts/run_gap_eval.py --tag gap-lexicon "
          f"--variants {args.out}")


if __name__ == "__main__":
    main()
