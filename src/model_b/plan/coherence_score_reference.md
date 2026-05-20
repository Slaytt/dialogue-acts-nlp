# Coherence Score for LDA — Reference & Rationale

## Which coherence metric to use?

**Recommended: C_v (also written CV or c_v)**

### Reference paper

> Röder, M., Both, A., & Hinneburg, A. (2015).
> **Exploring the Space of Topic Coherence Measures.**
> *Proceedings of the Eighth ACM International Conference on Web Search and Data Mining (WSDM '15)*, 399–408.
> https://doi.org/10.1145/2684822.2685324

This is the canonical evaluation paper. The authors systematically compare all existing coherence metrics against human judgements on a large set of LDA models and corpora.

---

## Summary of findings

The paper benchmarks four main families of coherence metrics:

| Metric   | Correlation with human judgement | Notes |
|----------|----------------------------------|-------|
| **C_v**  | **Highest (~0.85–0.90)**         | Uses NPMI (Normalized PMI) + sliding window of context; best overall |
| C_umass  | Moderate                         | Uses raw co-occurrence with document window; fast but noisy |
| C_uci    | Moderate                         | Uses PMI with external reference corpus; fragile on small corpora |
| C_npmi   | Good, slightly below C_v         | Normalized PMI, good but C_v still wins |

**C_v wins because:**
- It combines a sliding word context window (not the full document) with NPMI, which normalizes for word frequency bias.
- It is robust across corpus sizes, including medium corpora like Cornell Movie Dialogs (~300k utterances).
- It correlates most strongly with how humans perceive topic quality.

---

## Why not C_umass?

C_umass is fast and built into sklearn / gensim natively, but it only looks at raw document co-occurrence within training data. It is biased toward frequent words and correlates poorly with human judgement compared to C_v.

---

## Implementation note

Gensim's `CoherenceModel` supports C_v directly:

```python
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

coherence_model = CoherenceModel(
    topics=top_words_per_topic,   # list of list of strings
    texts=tokenized_texts,        # list of list of tokens (for sliding window)
    dictionary=gensim_dictionary,
    coherence='c_v'               # ← the metric
)
score = coherence_model.get_coherence()
```

Note: sklearn's `LatentDirichletAllocation` does not compute coherence. We extract the top-N words per topic from the sklearn model, then pass them to gensim's `CoherenceModel`.

---

## Practical k range to sweep

For the Cornell Movie Dialogs corpus (dialogue lines, short utterances):
- Minimum k: 5
- Maximum k: 25
- Step: 1 or 2

Typical sweet spots for dialogue corpora are k ∈ [8, 16]. The coherence curve usually peaks then plateaus or drops; pick the k at the first local maximum before the plateau.
