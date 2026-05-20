# Verification Checklist — Model-B LDA Improvement

Complete each item in order. Sign off with [x] and the observed value.

---

## Phase 1 — Coherence sweep (`find_optimal_k.py`)

- [ ] Script runs without error on the full corpus
- [ ] `resultats/coherence_vs_k.png` is generated and shows a visible peak or elbow
- [ ] `resultats/coherence_scores.csv` contains one row per k (5–25)
- [ ] Optimal k is printed and is in range [5, 25]
- [ ] C_v score at optimal k is ≥ 0.45 (below this threshold the topics are not meaningful)

Observed optimal k: ___  
Observed C_v score: ___

---

## Phase 2 — Model retraining (`extract_topics.py`)

- [ ] LDA retrained with optimal k using gensim (`alpha='auto'`, `eta='auto'`, `passes=10`)
- [ ] Model and dictionary saved to disk
- [ ] All utterances assigned a topic (`theme_dominant.isna().sum() == 0`)
- [ ] No single topic captures > 40% of all utterances (topic domination check)
- [ ] Distribution printed and looks balanced enough to be useful

Distribution (topic → % of utterances):

| Topic | % |
|-------|---|
|       |   |

---

## Phase 3 — Topic quality (manual)

For each topic, read the top-10 words and 10 sampled utterances. Mark as coherent (C) or incoherent (I).

| Topic | Top-3 words | Coherent? | Proposed label |
|-------|-------------|-----------|----------------|
| 0     |             |           |                |
| 1     |             |           |                |
| 2     |             |           |                |
| …     |             |           |                |

- [ ] All topics rated coherent (C)
- [ ] At most 1 "garbage" topic (catch-all for unclassifiable utterances) — acceptable
- [ ] Topic labels filled in `name_topics.py`

---

## Phase 4 — Topic diversity check

- [ ] No two topics share > 50% of their top-10 words
  - If this fails: k is too high; reduce by 2 and retrain

Pair with highest overlap: Topic ___ and Topic ___ → ___% shared words

---

## Phase 5 — Downstream impact (`analyser_csv.py`)

- [ ] `merge_results.py` re-run successfully (model-a + model-b outputs merged)
- [ ] `analyser_csv.py` re-run successfully
- [ ] Genre × topic heatmap saved to `resultats/themes_par_genre_nommes.png`
- [ ] The new distribution is visually different from the k=12 result (confirms improvement)
- [ ] No regression in model-a outputs (intent distribution unchanged)

---

## Final sign-off

- [ ] Optimal k is justified by C_v coherence (not hardcoded)
- [ ] All automated checks passed
- [ ] All topics are interpretable and labelled
- [ ] Results re-exported and plots updated
