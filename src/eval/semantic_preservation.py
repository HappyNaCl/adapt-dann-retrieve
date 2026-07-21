"""Semantic-preservation scoring for informal query variants (Phase 2).

An informal rewrite only keeps the formal qrels valid if it preserves the
information need. This scores that per query as the cosine between the formal and
informal query embeddings (both L2-normalized, so cosine == dot product). Low
cosine flags a drift suspect — a rewrite that may have changed the intent, whose
relevance judgements can no longer be trusted.

Cosine here uses the retrieval encoder (E5) by default, which is the same space
the gap is measured in; pass a different model (e.g. sentence-transformers/LaBSE)
for a less circular, translation-invariant signal. This is intentionally a
lightweight filter — 150-1000 short pairs encode in seconds on CPU.
"""
import numpy as np
from sentence_transformers import SentenceTransformer


def _encode(model: SentenceTransformer, texts: list[str], prefix: str, batch_size: int) -> np.ndarray:
    return np.asarray(
        model.encode(
            [prefix + t for t in texts],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    ).astype(np.float32)


def preservation_scores(
    formal: dict[str, str],
    variant: dict[str, str],
    model: SentenceTransformer,
    query_prefix: str = "",
    batch_size: int = 32,
) -> dict[str, float]:
    """{qid: cosine(formal, variant)} over the shared query ids.

    Missing ids on either side are skipped. Embeddings are L2-normalized, so the
    per-row dot product is the cosine similarity. Self-contained (no ranx/harness
    dependency) so it runs anywhere the encoder does.
    """
    qids = [q for q in formal if q in variant]
    fe = _encode(model, [formal[q] for q in qids], query_prefix, batch_size)
    ve = _encode(model, [variant[q] for q in qids], query_prefix, batch_size)
    cos = np.sum(fe * ve, axis=1)
    return {q: float(c) for q, c in zip(qids, cos)}


def summarize(scores: dict[str, float], threshold: float = 0.85) -> dict:
    """Distribution stats plus the drift suspects (cosine < threshold)."""
    vals = np.array(list(scores.values()))
    suspects = sorted((q for q, c in scores.items() if c < threshold), key=lambda q: scores[q])
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()) if vals.size else float("nan"),
        "p10": float(np.percentile(vals, 10)) if vals.size else float("nan"),
        "p25": float(np.percentile(vals, 25)) if vals.size else float("nan"),
        "min": float(vals.min()) if vals.size else float("nan"),
        "threshold": threshold,
        "n_below": len(suspects),
        "pct_below": round(100 * len(suspects) / vals.size, 1) if vals.size else 0.0,
        "suspects": suspects,
    }
