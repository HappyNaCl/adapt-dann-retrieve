"""Phase 1 gap decomposition: formal vs informal vs informal-normalized retrieval.

Requires the full-corpus embedding cache from run_eval.py (reuses it; only
queries are encoded here) and a pilot TSV with the `informal` column filled
(see make_pilot_set.py).

    python scripts/run_gap_eval.py --pilot experiments/pilot/pilot_queries.tsv --tag gap-pilot

Outputs:
  - per-variant rows appended to experiments/results.csv
  - experiments/gap/<tag>_per_query.csv   (paired per-query metrics, for figures)
  - experiments/gap/<tag>_summary.json    (gaps, decomposition, significance, cosines)
"""
import os
import sys

# ir_datasets opens MIRACL TSVs with the locale codepage (cp1252 on Windows)
# unless Python runs in UTF-8 mode, crashing on Indonesian text. Re-exec in
# UTF-8 mode before the heavy imports.
if sys.platform == "win32" and sys.flags.utf8_mode == 0:
    import subprocess

    sys.exit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))

import argparse
import csv
import json

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import miracl, normalize
from src.eval import harness, significance
from src.utils import log_result, set_seed

VARIANT_PAIRS = [  # (better-expected, worse-expected) comparisons to test
    ("formal", "informal"),
    ("normalized", "informal"),
    ("formal", "normalized"),
]


def load_pilot(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    filled = {r["query_id"]: r["informal"].strip() for r in rows if r["informal"].strip()}
    if not filled:
        sys.exit(f"[gap] no filled 'informal' rows in {path} — rewrite queries first")
    if len(filled) < len(rows):
        print(f"[gap] WARNING: {len(rows) - len(filled)} of {len(rows)} rows still empty; using {len(filled)}")
    return filled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--pilot", default="experiments/pilot/pilot_queries.tsv")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])

    model_name = args.model or cfg["backbone"]["name"]
    device = harness.resolve_device(args.device or cfg["backbone"].get("device", "auto"))
    batch_size = args.batch_size or cfg["backbone"]["batch_size"]
    split = cfg["data"]["split"]
    metrics = tuple(cfg["eval"]["metrics"])

    emb_dir = os.path.join(
        cfg["paths"]["embeddings_dir"], model_name.replace("/", "__"), f"{split}-full"
    )
    if not os.path.isdir(emb_dir) or not any(f.endswith(".npy") for f in os.listdir(emb_dir)):
        sys.exit(f"[gap] no full-corpus embedding cache at {emb_dir} — run run_eval.py (no --subset) first")

    informal = load_pilot(args.pilot)
    formal_all = miracl.load_queries(split)
    missing = sorted(set(informal) - set(formal_all))
    if missing:
        sys.exit(f"[gap] pilot qids not in {split} queries: {missing[:5]}...")
    qids = sorted(informal)
    variants = {
        "formal": {q: formal_all[q] for q in qids},
        "informal": informal,
        "normalized": normalize.normalize_queries(informal),
    }
    qrels_all = miracl.load_qrels(split)
    qrels = {q: qrels_all[q] for q in qids}
    print(f"[gap] {len(qids)} paired queries, model={model_name}, device={device}", flush=True)

    model = harness.load_model(model_name, device=device)
    per_query: dict[str, dict[str, dict[str, float]]] = {}  # variant -> metric -> qid -> score
    q_embs: dict[str, np.ndarray] = {}
    for name, queries in variants.items():
        vqids, q_emb = harness.encode_queries(
            model, queries, query_prefix=cfg["backbone"]["query_prefix"], batch_size=batch_size
        )
        q_embs[name] = q_emb
        scores, ids = harness.search(emb_dir, q_emb, k=cfg["eval"]["top_k"])
        run = harness.to_run(vqids, scores, ids)
        agg = harness.evaluate_run(run, qrels, metrics)
        per_query[name] = harness.evaluate_per_query(run, qrels, metrics)
        print(f"[result] {name}: " + " ".join(f"{k}={v:.4f}" for k, v in agg.items()), flush=True)
        log_result(
            cfg["paths"]["results_csv"],
            {
                "tag": f"{args.tag}-{name}",
                "model": model_name,
                "dataset": f"{cfg['data']['dataset']}/{split}",
                "corpus": "full",
                "n_docs": "",
                "n_queries": len(qids),
                "seed": cfg["seed"],
                **{k: round(v, 4) for k, v in agg.items()},
                "device": device,
            },
        )

    # Same-intent embedding cosines: direct evidence of register shift (paper figure).
    cosines = {
        pair: (q_embs[a] * q_embs[b]).sum(axis=1)  # embeddings are L2-normalized
        for pair, (a, b) in {
            "formal_vs_informal": ("formal", "informal"),
            "formal_vs_normalized": ("formal", "normalized"),
        }.items()
    }

    primary = metrics[0]  # ndcg@10 — decomposition is stated on the primary metric
    mean = {v: float(np.mean([per_query[v][primary][q] for q in qids])) for v in variants}
    total_gap = mean["formal"] - mean["informal"]
    vocab_part = mean["normalized"] - mean["informal"]
    residual = mean["formal"] - mean["normalized"]

    sig: dict[str, dict] = {}
    for a, b in VARIANT_PAIRS:
        p_values = {}
        for m in metrics:
            av = np.array([per_query[a][m][q] for q in qids])
            bv = np.array([per_query[b][m][q] for q in qids])
            t, p_t = significance.paired_t(av, bv)
            p_perm = significance.permutation_test(av, bv, seed=cfg["seed"])
            p_values[m] = {"t": t, "p_t": p_t, "p_perm": p_perm, "delta": float((av - bv).mean())}
        holm = significance.holm_bonferroni({m: v["p_perm"] for m, v in p_values.items()})
        sig[f"{a}_vs_{b}"] = {"metrics": p_values, "significant_holm": holm}

    summary = {
        "tag": args.tag,
        "model": model_name,
        "n_queries": len(qids),
        "mean": {v: {m: float(np.mean(list(per_query[v][m].values()))) for m in metrics} for v in variants},
        "decomposition": {
            "primary_metric": primary,
            "total_gap": total_gap,
            "vocabulary_part": vocab_part,
            "register_residual": residual,
            "vocabulary_pct_of_gap": 100 * vocab_part / total_gap if total_gap else float("nan"),
            "residual_pct_of_gap": 100 * residual / total_gap if total_gap else float("nan"),
        },
        "significance": sig,
        "same_intent_cosine": {
            name: {
                "mean": float(c.mean()),
                "p10": float(np.percentile(c, 10)),
                "p50": float(np.percentile(c, 50)),
                "p90": float(np.percentile(c, 90)),
            }
            for name, c in cosines.items()
        },
    }

    out_dir = "experiments/gap"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{args.tag}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, f"{args.tag}_per_query.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id"] + [f"{v}_{m}" for v in variants for m in metrics]
                        + list(cosines))
        for i, q in enumerate(qids):
            writer.writerow(
                [q] + [round(per_query[v][m][q], 4) for v in variants for m in metrics]
                + [round(float(c[i]), 4) for c in cosines.values()]
            )

    print(f"\n[gap] {primary}: formal={mean['formal']:.4f} informal={mean['informal']:.4f} "
          f"normalized={mean['normalized']:.4f}")
    print(f"[gap] total gap={total_gap:.4f} | vocabulary={vocab_part:.4f} "
          f"({summary['decomposition']['vocabulary_pct_of_gap']:.0f}%) | register residual={residual:.4f} "
          f"({summary['decomposition']['residual_pct_of_gap']:.0f}%)")
    print(f"[gap] summary -> {out_dir}/{args.tag}_summary.json")


if __name__ == "__main__":
    main()
