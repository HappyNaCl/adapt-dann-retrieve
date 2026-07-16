"""Dense-retrieval evaluation harness — the most reused code in the project.

Pipeline: encode corpus once (sharded, resumable, cached on disk) -> encode
queries -> streaming exact top-k search -> ranx metrics against qrels.

Corpus embeddings are cached per (model, corpus_tag) under results/embeddings/,
so later phases (informal query variants, aligned encoders) re-encode only
queries unless the document encoder itself changed.
"""
import json
import os
import time
from collections.abc import Iterator

import numpy as np
import torch
from ranx import Qrels, Run, evaluate as ranx_evaluate
from sentence_transformers import SentenceTransformer


def resolve_device(device: str = "auto") -> str:
    """'auto' -> cuda if available, else cpu. Explicit values pass through."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_name: str, device: str = "auto") -> SentenceTransformer:
    return SentenceTransformer(model_name, device=resolve_device(device))


def encode_corpus(
    model: SentenceTransformer,
    corpus_iter: Iterator[tuple[str, str]],
    out_dir: str,
    passage_prefix: str = "",
    batch_size: int = 32,
    shard_size: int = 25_000,
    log_every_shard: bool = True,
) -> None:
    """Encode (doc_id, text) pairs into float16 shards: shard_XXXXX.npy + .ids.json.

    Resumable: existing complete shards are skipped (the corpus iterator must be
    deterministic, which ir_datasets' docs_iter is). A shard is complete only if
    both its .npy and .ids.json exist; partial writes go to .tmp first.
    """
    os.makedirs(out_dir, exist_ok=True)
    shard_idx = 0
    buf_ids: list[str] = []
    buf_texts: list[str] = []
    t0 = time.time()

    def shard_paths(i: int) -> tuple[str, str]:
        return (
            os.path.join(out_dir, f"shard_{i:05d}.npy"),
            os.path.join(out_dir, f"shard_{i:05d}.ids.json"),
        )

    def flush() -> None:
        nonlocal shard_idx, buf_ids, buf_texts
        if not buf_ids:
            return
        npy_path, ids_path = shard_paths(shard_idx)
        if not (os.path.exists(npy_path) and os.path.exists(ids_path)):
            emb = np.asarray(
                model.encode(
                    [passage_prefix + t for t in buf_texts],
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            ).astype(np.float16)
            np.save(npy_path + ".tmp.npy", emb)
            os.replace(npy_path + ".tmp.npy", npy_path)
            with open(ids_path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(buf_ids, f)
            os.replace(ids_path + ".tmp", ids_path)
            if log_every_shard:
                done = (shard_idx + 1) * shard_size
                print(f"[encode] shard {shard_idx} done, ~{done} docs, {time.time() - t0:.0f}s elapsed", flush=True)
        shard_idx += 1
        buf_ids, buf_texts = [], []

    for doc_id, text in corpus_iter:
        buf_ids.append(doc_id)
        buf_texts.append(text)
        if len(buf_ids) >= shard_size:
            flush()
    flush()


def encode_queries(
    model: SentenceTransformer,
    queries: dict[str, str],
    query_prefix: str = "",
    batch_size: int = 32,
) -> tuple[list[str], np.ndarray]:
    qids = list(queries.keys())
    emb = np.asarray(
        model.encode(
            [query_prefix + queries[q] for q in qids],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    ).astype(np.float32)
    return qids, emb


def search(emb_dir: str, query_emb: np.ndarray, k: int = 100) -> tuple[np.ndarray, list[list[str]]]:
    """Streaming exact inner-product top-k over all shards in emb_dir.

    Never holds more than one shard in memory, so it works with the full
    1.45M-passage corpus on a small machine. Returns (scores[nq,k], doc_ids[nq][k]).
    """
    shard_files = sorted(f for f in os.listdir(emb_dir) if f.endswith(".npy") and ".tmp" not in f)
    if not shard_files:
        raise FileNotFoundError(f"no embedding shards in {emb_dir}")

    nq = query_emb.shape[0]
    top_scores = np.full((nq, k), -np.inf, dtype=np.float32)
    top_ids: list[list[str]] = [[""] * k for _ in range(nq)]

    for fname in shard_files:
        shard = np.load(os.path.join(emb_dir, fname)).astype(np.float32)
        with open(os.path.join(emb_dir, fname.replace(".npy", ".ids.json")), encoding="utf-8") as f:
            ids = json.load(f)
        sims = query_emb @ shard.T  # [nq, shard_size]

        kk = min(k, sims.shape[1])
        part = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        for qi in range(nq):
            cand_scores = np.concatenate([top_scores[qi], sims[qi, part[qi]]])
            cand_ids = top_ids[qi] + [ids[j] for j in part[qi]]
            order = np.argsort(-cand_scores)[:k]
            top_scores[qi] = cand_scores[order]
            top_ids[qi] = [cand_ids[j] for j in order]

    return top_scores, top_ids


def to_run(qids: list[str], top_scores: np.ndarray, top_ids: list[list[str]]) -> dict[str, dict[str, float]]:
    return {
        qid: {did: float(s) for did, s in zip(top_ids[qi], top_scores[qi]) if did}
        for qi, qid in enumerate(qids)
    }


def evaluate_run(
    run_dict: dict[str, dict[str, float]],
    qrels_dict: dict[str, dict[str, int]],
    metrics: tuple[str, ...] = ("ndcg@10", "recall@100", "mrr@10"),
) -> dict[str, float]:
    result = ranx_evaluate(Qrels(qrels_dict), Run(run_dict), list(metrics))
    if isinstance(result, dict):
        return {str(k): float(v) for k, v in result.items()}
    return {metrics[0]: float(result)}  # ranx returns a bare float for a single metric
