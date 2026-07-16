# ADAPT-DANN-Retrieve — Central Hypothesis (Phase 0, frozen 2026-07-16)

## Research question

Does **register-invariant representation alignment** (adversarial adapter on the
query encoder) close the formal↔colloquial retrieval gap in Indonesian better
than the established alternatives — input-side **query normalization/rewriting**
and representation-side **GPL / contrastive learning on pseudo-pairs**?

## Central hypothesis (H1)

On MIRACL-id with colloquial-register queries, adversarial register alignment
(DAPT + adversarial adapter, Route A) yields a statistically significant
improvement in nDCG@10 over both (a) DAPT + GPL and (b) DAPT +
contrastive-on-pseudo-pairs, while formal-query performance is not
significantly degraded.

## Null hypothesis (H0) — must be rejectable or acceptable

Adversarial alignment gives **no gain** over DAPT + GPL/contrastive: any
improvement on colloquial queries is attributable to domain-adaptive
pretraining and/or correspondence-based training, not to the adversarial
objective. **We commit to publishing either outcome** (positive result,
"simpler correspondence learning suffices", or the negative-result analysis of
why marginal alignment fails for retrieval).

## Frozen decisions (do not change mid-project)

| Decision     | Choice                                                                                                                            |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| Backbone     | `intfloat/multilingual-e5-base` (primary); IndoBERTweet as ablation base                                                          |
| Collection   | MIRACL-id dev: 960 queries, 9,668 qrels, 1,446,315 passages (corpus + qrels fixed forever; only queries get register-transformed) |
| Domain scope | **General** (all of MIRACL-id) — safer for a retrieval paper than a focused domain                                                |
| Metrics      | nDCG@10 (headline), Recall@100, MRR@10                                                                                            |
| Significance | Paired t-test + 100k-permutation randomization, Holm–Bonferroni across metrics, per-query deltas reported                         |
| Seed         | 42 (logged with every run in `experiments/results.csv`)                                                                           |

## Phase 0 gate

Harness must reproduce the published zero-shot number for the backbone:
**mE5-base, MIRACL-id dev = 51.1 nDCG@10 / 87.4 Recall@100**
(mE5 technical report, Wang et al. 2024, arXiv:2402.05672, Table 6; tolerance ±1 nDCG).

Secondary anchors: mE5-small 50.7, mE5-large 52.9 nDCG@10 (same table);
BM25 44.3 nDCG@10 (MIRACL paper).

## Known infrastructure constraint

The current machine is **CPU-only** (Quadro K620 2GB is unusable for modern
PyTorch). Phase 0–2 (harness, gap decomposition, test-set construction) are
CPU-feasible; corpus encoding is cached and shard-resumable to amortize the
cost. Phases 3–5 (DAPT, adversarial/GPL/contrastive training) will need a GPU
(Colab/Kaggle/cloud) — plan this before Phase 3.
