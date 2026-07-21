"""Check semantic preservation of an informal query variant (Phase 2 validation).

Scores cosine(formal, variant) per query and flags drift suspects — rewrites that
may have changed the information need, whose reuse of the formal qrels is unsafe.

    python scripts/check_preservation.py --variants experiments/pilot/pilot_informal_llm.tsv --column llm
    python scripts/check_preservation.py --column informal   # the pilot's manual arm

Formal text comes from the pilot's `formal` column; the variant is any column in
--variants (default file: the pilot itself, so --column informal works with no
--variants). Writes a per-query CSV sorted worst-first and prints the suspects.
"""
import os
import sys

if sys.platform == "win32" and sys.flags.utf8_mode == 0:
    import subprocess

    sys.exit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))

import argparse
import csv

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval import semantic_preservation


def load_column(path: str, column: str) -> dict[str, str]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows or column not in rows[0]:
        sys.exit(f"[preserve] {path} has no '{column}' column (has: {list(rows[0].keys()) if rows else '—'})")
    return {r["query_id"]: r[column].strip() for r in rows if r.get(column, "").strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--pilot", default="experiments/pilot/pilot_queries.tsv", help="source of the 'formal' column")
    ap.add_argument("--variants", default=None, help="TSV holding the variant column (default: --pilot)")
    ap.add_argument("--column", required=True, help="variant column to score, e.g. llm / informal")
    ap.add_argument("--model", default=None, help="encoder (default: backbone; try LaBSE for a less circular signal)")
    ap.add_argument("--threshold", type=float, default=0.85, help="cosine below this = drift suspect")
    ap.add_argument("--out", default=None, help="per-query CSV (default: experiments/gap/preserve_<column>.csv)")
    ap.add_argument("--show", type=int, default=15, help="how many worst suspects to print")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    formal = load_column(args.pilot, "formal")
    variant = load_column(args.variants or args.pilot, args.column)

    from sentence_transformers import SentenceTransformer

    model_name = args.model or cfg["backbone"]["name"]
    prefix = cfg["backbone"].get("query_prefix", "") if not args.model else ""
    print(f"[preserve] scoring '{args.column}' vs formal with {model_name} ({len(variant)} queries)...")
    device = cfg["backbone"].get("device", "auto")
    device = "cpu" if device == "auto" else device
    model = SentenceTransformer(model_name, device=device)
    scores = semantic_preservation.preservation_scores(formal, variant, model, prefix, cfg["backbone"]["batch_size"])
    stats = semantic_preservation.summarize(scores, args.threshold)

    out = args.out or f"experiments/gap/preserve_{args.column}.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "cosine", "formal", args.column])
        for q in sorted(scores, key=lambda q: scores[q]):
            w.writerow([q, f"{scores[q]:.4f}", formal[q], variant[q]])

    print(f"[preserve] mean={stats['mean']:.4f}  p10={stats['p10']:.4f}  p25={stats['p25']:.4f}  min={stats['min']:.4f}")
    print(f"[preserve] {stats['n_below']}/{stats['n']} ({stats['pct_below']}%) below cosine {args.threshold} — drift suspects")
    print(f"[preserve] full per-query scores -> {out}\n")
    print(f"[preserve] worst {min(args.show, len(stats['suspects']))} suspects:")
    for q in stats["suspects"][: args.show]:
        print(f"  {scores[q]:.3f}  {q}")
        print(f"        F: {formal[q]}")
        print(f"        {args.column[:1].upper()}: {variant[q]}")


if __name__ == "__main__":
    main()
