# Model-B Improvement Plan — LDA Topic Extraction

## Goal

Replace the hardcoded `N_THEMES = 12` with a data-driven optimal k found via C_v coherence scoring, and harden the rest of the pipeline (vectorizer tuning, topic naming, verification).

---

## Current state

- `src/model_b/extract_topics.py`
- Uses `sklearn.decomposition.LatentDirichletAllocation`
- k is fixed at 12, chosen without empirical justification
- No coherence evaluation at all
- Custom stop-word list is hand-tuned and includes first names (fragile)

---

## Step 1 — Add C_v coherence sweep

**File to create:** `src/model_b/find_optimal_k.py`

### What it does

1. Load and clean the corpus (reuse `nettoyer_repliques`).
2. Build a gensim `Dictionary` and `Corpus` from the same tokenized texts used by sklearn (ensures consistency).
3. Train one LDA model per k in range [5, 25] (step 1).
4. For each k, compute C_v coherence via `gensim.models.CoherenceModel`.
5. Plot coherence score vs k (`matplotlib`), save to `resultats/coherence_vs_k.png`.
6. Print the optimal k (argmax of the curve) and the score table.

### Key code sketch

```python
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

K_RANGE = range(5, 26)
TOP_WORDS = 15   # same as N_MOTS_PAR_THEME

tokenized = [text.split() for text in textes_nettoyes]
dictionary = Dictionary(tokenized)
dictionary.filter_extremes(no_below=10, no_above=0.95)
corpus = [dictionary.doc2bow(doc) for doc in tokenized]

scores = []
for k in K_RANGE:
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k,
                   random_state=42, passes=10, alpha='auto', eta='auto')
    cm = CoherenceModel(model=lda, texts=tokenized,
                        dictionary=dictionary, coherence='c_v')
    scores.append((k, cm.get_coherence()))

optimal_k = max(scores, key=lambda x: x[1])[0]
```

### Output

- `resultats/coherence_vs_k.png` — the elbow/peak plot
- Console table with k → C_v score
- `resultats/coherence_scores.csv` — raw numbers for reproducibility

---

## Step 2 — Switch to gensim LDA in the main pipeline

**File to modify:** `src/model_b/extract_topics.py`

### Why

Gensim's `LdaModel` supports `alpha='auto'` and `eta='auto'` (asymmetric Dirichlet priors learned from data), which consistently outperforms sklearn's fixed symmetric priors for short-text corpora like dialogue lines.

### Changes

| Current (sklearn) | New (gensim) |
|---|---|
| `CountVectorizer` → sparse matrix | `Dictionary` + `doc2bow` corpus |
| `LatentDirichletAllocation(n_components=k)` | `LdaModel(num_topics=k, alpha='auto', eta='auto', passes=10)` |
| `joblib.dump` | `lda.save(path)` / `LdaModel.load(path)` |
| `lda.transform(X).argmax(axis=1)` | `lda[corpus]` → parse distribution |

`N_THEMES` becomes a parameter loaded from `resultats/coherence_scores.csv` or passed via CLI.

---

## Step 3 — Vectorizer / preprocessing tuning

**File to modify:** `src/model_b/extract_topics.py` (preprocessing section)

1. **Stop-words**: replace the brittle hand-written list with a union of `spacy`'s `STOP_WORDS` set + a small dialogue-specific addendum (interjections only). First names do not belong here — they can be topic-informative (character names signal certain scenes).
2. **min_df**: currently 10. Lower to 5 if corpus size allows (more vocabulary diversity for coherence).
3. **max_features**: currently 5000. Set to `None` after filtering; let `min_df` / `max_df` do the work.
4. **Token pattern**: `[a-z]{3,}` is fine; keep it.

---

## Step 4 — Topic naming (interpretability)

**File to create:** `src/model_b/name_topics.py`

After finding the optimal k:
1. Print top-20 words per topic.
2. Provide a mapping dict `TOPIC_NAMES = {0: "...", 1: "...", ...}` that a human fills in based on inspection.
3. The mapping is applied in `extract_topics.py` to produce a `topic_label` column alongside `theme_dominant`.

This is a manual but documented step — the plan makes it explicit so it is not skipped.

---

## Step 5 — Verification

**File to create:** `src/model_b/plan/verification_checklist.md` (see companion file)

### Automated checks

| Check | How |
|---|---|
| C_v score of chosen k ≥ 0.45 | Assert in `find_optimal_k.py` |
| No topic has > 40% of documents | Distribution check in `extract_topics.py` |
| No two topics share > 50% of their top-10 words | Topic diversity check post-training |
| All utterances get a topic | Assert `theme_dominant.isna().sum() == 0` |

### Manual checks

1. Read top-10 words of each topic — they should be interpretable as a coherent theme.
2. Sample 10 utterances per topic — they should feel thematically consistent.
3. Run `src/analysis/analyser_csv.py` — genre × topic distributions should shift meaningfully from current results (proving the new k is not equivalent to k=12).

---

## Execution order

```
1. find_optimal_k.py          → determines best k, saves plot + CSV
2. extract_topics.py          → retrain with best k, save model
3. name_topics.py             → inspect + name topics (manual)
4. merge_results.py           → re-merge with model-a outputs
5. analyser_csv.py            → re-run cross-analyses
6. verification_checklist.md  → sign off on each check
```

---

## Files created / modified summary

| File | Action |
|---|---|
| `src/model_b/find_optimal_k.py` | CREATE — coherence sweep |
| `src/model_b/extract_topics.py` | MODIFY — switch to gensim, use optimal k |
| `src/model_b/name_topics.py` | CREATE — topic labelling helper |
| `src/model_b/plan/coherence_score_reference.md` | CREATE — metric justification |
| `src/model_b/plan/improvement_plan.md` | CREATE — this file |
| `src/model_b/plan/verification_checklist.md` | CREATE — sign-off checklist |
| `resultats/coherence_vs_k.png` | GENERATED output |
| `resultats/coherence_scores.csv` | GENERATED output |
