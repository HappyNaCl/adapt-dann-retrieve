"""Zero-shot dense-retrieval evaluation on MIRACL-id.

Full corpus (Phase 0 sanity check):
    python scripts/run_eval.py --config configs/base.yaml --tag e5-base-zeroshot

Smoke test on a subset (all qrels-relevant docs + N distractors; numbers are
NOT comparable to published full-corpus results — pipeline validation only):
    python scripts/run_eval.py --config configs/base.yaml --subset 20000 --tag smoke
"""
import argparse
import itertools
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import miracl
from src.eval import harness
from src.utils import log_result, set_seed


def build_subset_corpus(qrels: dict[str, dict[str, int]], n_distractors: int, split: str):
    """qrels-complete subset: every judged doc + the first N other docs."""
    judged = {did for docs in qrels.values() for did in docs}
    pairs = [(did, text) for did, text in miracl.fetch_docs(sorted(judged), split=split).items()]
    distractors = itertools.islice(
        ((did, t) for did, t in miracl.iter_corpus(split) if did not in judged), n_distractors
    )
    return itertools.chain(pairs, distractors), len(judged) + n_distractors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--tag", required=True, help="run name for the results log")
    ap.add_argument("--subset", type=int, default=0, help="if >0, smoke-test on judged docs + N distractors")
    ap.add_argument("--model", default=None, help="override backbone from config")
    ap.add_argument("--device", default=None, help="override device: auto | cpu | cuda | cuda:N")
    ap.add_argument("--batch-size", type=int, default=None, help="override encode batch size (raise on GPU)")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])

    model_name = args.model or cfg["backbone"]["name"]
    device = harness.resolve_device(args.device or cfg["backbone"].get("device", "auto"))
    batch_size = args.batch_size or cfg["backbone"]["batch_size"]
    split = cfg["data"]["split"]
    print(f"[run] model={model_name} split={split} subset={args.subset or 'FULL'} "
          f"device={device} batch_size={batch_size}", flush=True)

    queries = miracl.load_queries(split)
    qrels = miracl.load_qrels(split)
    print(f"[run] {len(queries)} queries, {len(qrels)} qrels topics", flush=True)

    model_slug = model_name.replace("/", "__")
    corpus_tag = f"subset{args.subset}" if args.subset else "full"
    emb_dir = os.path.join(cfg["paths"]["embeddings_dir"], model_slug, f"{split}-{corpus_tag}")

    if args.subset:
        corpus_iter, n_docs = build_subset_corpus(qrels, args.subset, split)
    else:
        corpus_iter, n_docs = miracl.iter_corpus(split), miracl.corpus_size(split)

    model = harness.load_model(model_name, device=device)

    t0 = time.time()
    harness.encode_corpus(
        model,
        corpus_iter,
        emb_dir,
        passage_prefix=cfg["backbone"]["passage_prefix"],
        batch_size=batch_size,
        shard_size=cfg["encode"]["shard_size"],
    )
    encode_s = time.time() - t0
    print(f"[run] corpus encode: {encode_s:.0f}s for {n_docs} docs", flush=True)

    qids, q_emb = harness.encode_queries(
        model, queries, query_prefix=cfg["backbone"]["query_prefix"], batch_size=batch_size
    )
    scores, ids = harness.search(emb_dir, q_emb, k=cfg["eval"]["top_k"])
    run = harness.to_run(qids, scores, ids)
    metrics = harness.evaluate_run(run, qrels, tuple(cfg["eval"]["metrics"]))

    print(f"[result] tag={args.tag} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)
    log_result(
        cfg["paths"]["results_csv"],
        {
            "tag": args.tag,
            "model": model_name,
            "dataset": f"{cfg['data']['dataset']}/{split}",
            "corpus": corpus_tag,
            "n_docs": n_docs,
            "n_queries": len(queries),
            "seed": cfg["seed"],
            "device": device,
            **{k: round(v, 4) for k, v in metrics.items()},
            "encode_seconds": round(encode_s),
        },
    )


if __name__ == "__main__":
    main()
