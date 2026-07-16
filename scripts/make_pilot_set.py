"""Sample MIRACL-id dev queries into a TSV for manual colloquial rewriting.

Phase 1 pilot (diagnostic, not the release test set): rewrite the `informal`
column into colloquial Indonesian — same information need, casual register
(slang, abbreviations, chat style). Keep the tab structure intact.

    python scripts/make_pilot_set.py --n 150
    -> experiments/pilot/pilot_queries.tsv
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
import random

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import miracl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--n", type=int, default=150, help="pilot size (CLAUDE.md: 100-200)")
    ap.add_argument("--out", default="experiments/pilot/pilot_queries.tsv")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    queries = miracl.load_queries(cfg["data"]["split"])
    qids = sorted(queries)
    random.Random(cfg["seed"]).shuffle(qids)
    picked = sorted(qids[: args.n])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["query_id", "formal", "informal"])
        for qid in picked:
            writer.writerow([qid, queries[qid], ""])

    print(f"[pilot] wrote {len(picked)} queries to {args.out} (seed={cfg['seed']})")
    print("[pilot] fill the 'informal' column with colloquial rewrites, then run scripts/run_gap_eval.py")


if __name__ == "__main__":
    main()
