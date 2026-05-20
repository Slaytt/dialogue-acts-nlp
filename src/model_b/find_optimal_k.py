import os
import sys
import csv
import time

import numpy as np
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.preprocessing.load_cornell import charger_cornell
from src.model_b.extract_topics import nettoyer_repliques

# ── sweep parameters ──────────────────────────────────────────────────────────
K_MIN = 5
K_MAX = 25
LDA_PASSES = 10
TOP_WORDS = 15       # words per topic used by CoherenceModel
RANDOM_STATE = 42
C_V_THRESHOLD = 0.45  # below this the topics are considered noise

CHEMIN_RESULTATS = os.path.join(os.path.dirname(__file__), '..', '..', 'resultats')


# ── corpus preparation ────────────────────────────────────────────────────────

def build_gensim_corpus(textes_nettoyes):
    """
    Tokenize cleaned strings and build a gensim Dictionary + BoW corpus.
    Applies the same frequency cuts as the sklearn vectorizer (min_df=5, max_df=0.95).
    Returns (tokenized, dictionary, corpus).
    """
    tokenized = [text.split() for text in textes_nettoyes]

    dictionary = Dictionary(tokenized)
    # filter_extremes mirrors CountVectorizer(min_df=5, max_df=0.95)
    dictionary.filter_extremes(no_below=5, no_above=0.95)
    dictionary.compactify()

    corpus = [dictionary.doc2bow(doc) for doc in tokenized]

    print(f"Dictionary : {len(dictionary):,} tokens")
    print(f"Corpus     : {len(corpus):,} documents")
    return tokenized, dictionary, corpus


# ── coherence sweep ───────────────────────────────────────────────────────────

def sweep_coherence(tokenized, dictionary, corpus, k_range):
    """
    Train one gensim LDA per k value, compute C_v coherence.
    Returns list of (k, c_v_score) sorted by k.
    """
    scores = []
    for k in k_range:
        t0 = time.time()
        print(f"  k={k:>2} — training... ", end="", flush=True)

        lda = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            random_state=RANDOM_STATE,
            passes=LDA_PASSES,
            alpha='auto',     # asymmetric prior, learned from data
            eta='auto',       # asymmetric prior, learned from data
            per_word_topics=False,
        )

        # Extract top words as strings for CoherenceModel
        top_words = [
            [w for w, _ in lda.show_topic(t, topn=TOP_WORDS)]
            for t in range(k)
        ]

        cm = CoherenceModel(
            topics=top_words,
            texts=tokenized,
            dictionary=dictionary,
            coherence='c_v',
        )
        score = cm.get_coherence()
        elapsed = time.time() - t0
        scores.append((k, score))
        print(f"C_v = {score:.4f}  ({elapsed:.0f}s)")

    return scores


# ── output helpers ────────────────────────────────────────────────────────────

def save_scores_csv(scores, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['k', 'c_v'])
        writer.writerows(scores)
    print(f"Scores CSV saved : {path}")


def plot_coherence(scores, path):
    ks = [s[0] for s in scores]
    cvs = [s[1] for s in scores]
    optimal_k = ks[int(np.argmax(cvs))]
    best_cv = max(cvs)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(ks, cvs, marker='o', linewidth=2, color='steelblue', label='C_v coherence')
    ax.axvline(x=optimal_k, color='tomato', linestyle='--', alpha=0.8,
               label=f'Optimal k={optimal_k}  (C_v={best_cv:.4f})')
    ax.scatter([optimal_k], [best_cv], color='tomato', zorder=5, s=90)
    ax.axhline(y=C_V_THRESHOLD, color='grey', linestyle=':', alpha=0.6,
               label=f'Threshold ({C_V_THRESHOLD})')

    ax.set_xlabel('Number of topics (k)', fontsize=12)
    ax.set_ylabel('C_v coherence score', fontsize=12)
    ax.set_title('LDA — C_v Coherence vs Number of Topics\n(Cornell Movie Dialogs, genre-known utterances)',
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ks)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot saved       : {path}")


def print_score_table(scores, optimal_k):
    print(f"\n{'k':>4}  {'C_v':>8}")
    print("-" * 18)
    for k, score in scores:
        tag = " ← OPTIMAL" if k == optimal_k else ""
        print(f"{k:>4}  {score:>8.4f}{tag}")


def verify(best_cv, optimal_k):
    print("\n--- Verification ---")
    if best_cv < C_V_THRESHOLD:
        print(f"WARNING: best C_v ({best_cv:.4f}) is below threshold ({C_V_THRESHOLD}).")
        print("Topics may not be meaningful. Consider adjusting preprocessing.")
    else:
        print(f"C_v = {best_cv:.4f} >= {C_V_THRESHOLD}  — topics look meaningful.")
    print(f"Recommended N_THEMES = {optimal_k}")
    print("Update N_THEMES in extract_topics.py or pass it as an argument.")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== FIND OPTIMAL k — C_v COHERENCE SWEEP ===\n")

    df_cornell = charger_cornell()
    df_analyse = df_cornell[df_cornell["character_gender"].isin(["m", "f"])].copy()
    print(f"\nUtterances with known gender : {len(df_analyse):,}")

    textes_propres = nettoyer_repliques(df_analyse, col_texte="text")

    print("\nBuilding gensim corpus...")
    tokenized, dictionary, corpus = build_gensim_corpus(textes_propres)

    print(f"\n=== Coherence sweep  k={K_MIN}..{K_MAX}  passes={LDA_PASSES} ===\n")
    scores = sweep_coherence(tokenized, dictionary, corpus, range(K_MIN, K_MAX + 1))

    optimal_k, best_cv = max(scores, key=lambda x: x[1])

    print_score_table(scores, optimal_k)

    os.makedirs(CHEMIN_RESULTATS, exist_ok=True)
    save_scores_csv(scores, os.path.join(CHEMIN_RESULTATS, 'coherence_scores.csv'))
    plot_coherence(scores, os.path.join(CHEMIN_RESULTATS, 'coherence_vs_k.png'))

    verify(best_cv, optimal_k)

    print("\n=== Done ===")
