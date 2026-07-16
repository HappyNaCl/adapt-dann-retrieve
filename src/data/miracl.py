"""MIRACL-id access layer on top of ir_datasets.

Corpus and qrels stay fixed for the whole project; only queries get
transformed into colloquial register in later phases.
"""
from collections.abc import Iterator

import ir_datasets

DATASET = "miracl/id"


def load_dataset(split: str = "dev"):
    return ir_datasets.load(f"{DATASET}/{split}")


def load_queries(split: str = "dev") -> dict[str, str]:
    ds = load_dataset(split)
    return {q.query_id: q.text for q in ds.queries_iter()}


def load_qrels(split: str = "dev") -> dict[str, dict[str, int]]:
    """qrels as {qid: {doc_id: relevance}} — the format ranx expects."""
    ds = load_dataset(split)
    qrels: dict[str, dict[str, int]] = {}
    for qrel in ds.qrels_iter():
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
    return qrels


def doc_text(doc) -> str:
    """MIRACL passages carry a page title; prepend it, as in the MIRACL baselines."""
    title = (doc.title or "").strip()
    text = doc.text.strip()
    return f"{title} {text}".strip() if title else text


def iter_corpus(split: str = "dev") -> Iterator[tuple[str, str]]:
    """Deterministic (doc_id, text) iterator over the full 1.45M-passage corpus."""
    ds = load_dataset(split)
    for doc in ds.docs_iter():
        yield doc.doc_id, doc_text(doc)


def corpus_size(split: str = "dev") -> int:
    return load_dataset(split).docs_count()


def fetch_docs(doc_ids: list[str], split: str = "dev") -> dict[str, str]:
    """Random-access lookup, used to build qrels-complete corpus subsets."""
    store = load_dataset(split).docs_store()
    return {did: doc_text(store.get(did)) for did in doc_ids}
