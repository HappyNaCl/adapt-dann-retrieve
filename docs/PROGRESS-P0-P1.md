# Progress Report — Phases 0–1

**Project:** Register-invariant representation alignment for Indonesian dense retrieval
**Period:** Week 1 (July 16, 2026)
**Status:** Phase 0 gate ✅ passed · Phase 1 gate ⚠️ passed with a caveat (details below)

---

## Phase 0 — Scoping & evaluation infrastructure

**Goal:** lock all experimental choices and prove the evaluation pipeline is
trustworthy _before_ any modeling.

### Frozen decisions (configs/base.yaml — will not change mid-project)

| Choice       | Value                                                                    |
| ------------ | ------------------------------------------------------------------------ |
| Backbone     | `intfloat/multilingual-e5-base`                                          |
| Collection   | MIRACL-id (960 dev queries, 1.45M Wikipedia passages), via `ir_datasets` |
| Metrics      | nDCG@10 (primary), Recall@100, MRR@10                                    |
| Significance | paired t-test + 100k-permutation test, Holm–Bonferroni across metrics    |
| Seed         | 42 (all sampling and training reproducible)                              |

### Built

- **Dense-retrieval eval harness** (`src/eval/harness.py`): corpus encoded once
  into resumable float16 shards; streaming exact top-k search (runs on any
  machine, full corpus never held in memory); ranx metrics; every run appends
  to a central results CSV.
- Significance-testing module fixed in advance (`src/eval/significance.py`).
- MIRACL-id access layer; all results logged with model/corpus/seed/device.

### Validation gate

The harness must reproduce a published zero-shot number, otherwise every later
comparison is untrustworthy.

> **Full-corpus zero-shot: nDCG@10 = 0.5103, Recall@100 = 0.8738** — matches
> the published mE5-base result on MIRACL-id (≈ 0.51) within the ±1-point
> tolerance. **Gate passed.**

---

## Phase 1 — Decomposing the formal/informal gap before modeling

**Goal:** prove the problem exists, and split it into the part fixable by
vocabulary normalization vs the register/distribution-shift residual that only
representation alignment can address. The residual is the ceiling on the
proposed method's contribution — measured _before_ investing in the method.

### Built

- **Pilot informal query set** (`experiments/pilot/`): 150 seed-sampled
  MIRACL-id dev queries rewritten into colloquial Indonesian
  (manual + LLM-assisted, human-reviewed). Corpus and qrels stay fixed —
  only the query register changes, so relevance judgments remain valid.
- **Normalizer** (`src/data/normalize.py`): Kamus Alay lexicon substitution —
  two published lexicons merged (Salsabila et al. 2018; Ibrohim & Budi 2019;
  15.5k entries), with elongation handling ("bangetttt" → "banget").
- **Gap-decomposition eval** (`scripts/run_gap_eval.py`): three-way paired
  comparison reusing the cached corpus embeddings; outputs per-query scores,
  significance tests, and same-intent embedding cosines.

### Results (150 paired queries, full 1.45M-passage corpus)

| Query variant         | nDCG@10 | Recall@100 | MRR@10 |
| --------------------- | ------- | ---------- | ------ |
| formal (original)     | 0.5097  | 0.8813     | 0.6092 |
| informal (colloquial) | 0.4496  | 0.8111     | 0.5476 |
| informal → normalized | 0.4730  | 0.8312     | 0.5826 |

**Decomposition of the 6.0-point nDCG@10 gap:**

- **Total register gap: −6.0 nDCG@10** — significant on all three metrics
  after Holm–Bonferroni (permutation p = 0.0007). _The problem is real._
- **Vocabulary part: 2.3 points (39%)** — recovered by lexicon normalization,
  but not significant after correction at n = 150 (raw p ≈ 0.048).
- **Register residual: 3.7 points (61%) — statistically significant**
  (p = 0.016; Recall@100 residual is 5.0 points at p = 0.0001). Relevant
  documents fall out of the top-100 entirely — unfixable by input-side
  normalization.
- Embedding evidence: same-intent formal↔informal pairs average cosine 0.949;
  normalization only closes this to 0.960. A persistent embedding-space
  displacement remains beyond spelling differences.

### Gate interpretation & next step

**The majority (61%) of the retrieval gap is register shift, not vocabulary** —
this already contradicts the assumption behind normalization-only approaches
and is a presentable finding on its own.

The 3.7-point residual sits _below_ the pre-registered ~5-point pivot
threshold, with two known biases in opposite directions:

1. The lexicon normalizer is weak (maps slang to canonical-colloquial, not
   formal) → residual may be **overstated**.
2. The pilot rewrites are mild (light slang, no code-mixing) → residual may be
   **understated**.

**Planned next step (cheap — only queries are re-encoded):** bound the
residual from both sides before committing to the method phases — (a) a
stronger normalization arm (STIF-style seq2seq / LLM rewriting), (b) a
heavy-informal variant of the same 150 queries. If the residual survives
strong normalization at ≥3–4 points, the alignment method has a defensible
target; if not, the project reframes around DAPT + normalization, per the
pre-registered decision rule.
