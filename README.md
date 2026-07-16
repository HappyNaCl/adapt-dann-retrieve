# ADAPT-DANN-Retrieve

Register-invariant representation alignment for Indonesian dense retrieval:
does an adversarial adapter beat query normalization and GPL/contrastive
baselines on colloquial-register queries? See `CLAUDE.md` for the full roadmap
and `docs/HYPOTHESIS.md` for the frozen Phase 0 decisions.

## Setup

Python 3.10. On this machine use the `nlp` conda env; on a fresh (e.g. GPU)
machine:

```bash
# GPU box: install the CUDA torch build first, then the rest
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Device is auto-detected (`device: auto` in `configs/base.yaml`): CUDA when
available, CPU otherwise. Override per run with `--device cuda --batch-size 128`.

MIRACL-id is fetched automatically via `ir_datasets` (cached in
`~/.ir_datasets`):

```bash
python scripts/download_miracl.py
```

## Evaluation harness (Phase 0)

```bash
# smoke test: all judged docs + 20k distractors (pipeline validation only)
python scripts/run_eval.py --config configs/base.yaml --subset 20000 --tag smoke-e5base

# full-corpus zero-shot sanity check (gate: 51.1 nDCG@10 ±1, mE5 tech report)
python scripts/run_eval.py --config configs/base.yaml --tag e5-base-zeroshot
```

Corpus embeddings are cached as resumable float16 shards under
`results/embeddings/<model>/<corpus>/`; interrupted runs pick up at the last
complete shard. Results append to `experiments/results.csv`.

Phase 0 gate passed 2026-07-16: full-corpus zero-shot nDCG@10 = 0.5103
(published mE5-base ≈ 0.51). ✅

## Gap decomposition (Phase 1)

```bash
# 1. sample 150 dev queries into a TSV (seeded, reproducible)
python scripts/make_pilot_set.py --n 150

# 2. manually rewrite the 'informal' column of
#    experiments/pilot/pilot_queries.tsv into colloquial Indonesian

# 3. three-way eval (formal / informal / Kamus-Alay-normalized) against the
#    cached full-corpus embeddings — only queries are re-encoded
python scripts/run_gap_eval.py --tag gap-pilot
```

Prints the vocabulary-vs-register decomposition of the retrieval gap and
writes `experiments/gap/<tag>_summary.json` (means, paired t + permutation
significance with Holm-Bonferroni, same-intent embedding cosines) plus a
per-query CSV for figures. The register residual is the go/no-go number for
the alignment method (CLAUDE.md Phase 1 gate).

Normalization merges two lexicons in `resources/lexicons/` (~15.5k entries):
`new_kamusalay.csv` (Ibrohim & Budi 2019) and
`colloquial-indonesian-lexicon.csv` (Salsabila et al. 2018).

## Layout

```
configs/        frozen experiment configs (base.yaml = Phase 0 choices)
docs/           hypothesis one-pager, notes
resources/      Kamus Alay lexicons (committed; small)
scripts/        entry points (download, run_eval, make_pilot_set, run_gap_eval)
src/data/       MIRACL access layer, normalize.py (later: make_informal.py)
src/eval/       harness (encode/search/metrics), significance tests
src/models/     later: GRL, adapters
src/train/      later: DAPT, adversarial, GPL, contrastive
experiments/    results.csv, pilot query TSV, gap outputs, notebooks
results/        embeddings cache (gitignored)
```
