"""Download MIRACL-id (docs, queries, qrels) via ir_datasets and report counts.

ir_datasets caches under ~/.ir_datasets; iterating docs triggers the corpus
download on first run.
"""
import time

import ir_datasets


def main() -> None:
    ds = ir_datasets.load("miracl/id/dev")

    n_queries = sum(1 for _ in ds.queries_iter())
    n_qrels = sum(1 for _ in ds.qrels_iter())
    print(f"queries: {n_queries}", flush=True)
    print(f"qrels:   {n_qrels}", flush=True)

    t0 = time.time()
    n_docs = 0
    for _ in ds.docs_iter():
        n_docs += 1
        if n_docs % 200_000 == 0:
            print(f"  ...{n_docs} docs ({time.time() - t0:.0f}s)", flush=True)
    print(f"docs:    {n_docs} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
