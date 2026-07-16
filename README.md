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
`data/embeddings/<model>/<corpus>/`; interrupted runs pick up at the last
complete shard. Results append to `experiments/results.csv`.

## Layout

```
configs/        frozen experiment configs (base.yaml = Phase 0 choices)
docs/           hypothesis one-pager, notes
scripts/        entry points (download, run_eval)
src/data/       MIRACL access layer (later: normalize.py, make_informal.py)
src/eval/       harness (encode/search/metrics), significance tests
src/models/     later: GRL, adapters
src/train/      later: DAPT, adversarial, GPL, contrastive
experiments/    results.csv + analysis notebooks
data/           embeddings cache (gitignored)
```
