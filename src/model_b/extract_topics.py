# Modèle B — Extraction de thèmes par LDA (non-supervisé)
# Chaque réplique reçoit une distribution de probabilités sur les thèmes.

import os
import sys
import joblib
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.preprocessing.clean_text import nettoyer_texte

N_THEMES = 12
N_MOTS_PAR_THEME = 15
CHEMIN_MODELE_LDA = os.path.join(os.path.dirname(__file__), "modele_lda.joblib")
CHEMIN_VECTORISEUR = os.path.join(os.path.dirname(__file__), "vectoriseur_lda.joblib")


def nettoyer_repliques(df, col_texte="text"):
    """Nettoie les répliques avec spaCy pour le LDA. Retourne une Series sans les vides."""
    print("Chargement du modèle spaCy...")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    print(f"Nettoyage de {len(df)} répliques avec spaCy...")
    textes_nettoyes = df[col_texte].apply(lambda t: nettoyer_texte(str(t), nlp))

    nb_avant = len(textes_nettoyes)
    textes_nettoyes = textes_nettoyes[textes_nettoyes.str.strip() != ""]
    print(f"Répliques supprimées (vides après nettoyage) : {nb_avant - len(textes_nettoyes)}")

    return textes_nettoyes


def entrainer_lda(textes_nettoyes, n_themes=N_THEMES):
    """Entraîne un modèle LDA. Retourne (modele_lda, vectoriseur, matrice_comptes)."""
    # Stop words spécifiques au LDA : interjections + verbes génériques + prénoms courants
    STOP_WORDS_CUSTOM = [
        "uh", "huh", "ah", "yeah", "oh", "hey", "gonna", "wanna", "gotta",
        "um", "hmm", "yep", "nah", "ok", "okay", "alright", "wow", "ooh",
        "get", "got", "go", "come", "know", "want", "say", "tell", "think",
        "make", "take", "let", "look", "need", "mean", "use", "try", "ask",
        "seem", "give", "keep", "happen",
        "thing", "way", "right", "good", "sure", "well", "fine", "great",
        "lot", "kind", "maybe", "probably", "actually", "really", "still",
        "john", "jack", "george", "bob", "tom", "joe", "chris",
    ]

    print("\nVectorisation (CountVectorizer)...")
    vectoriseur = CountVectorizer(
        max_df=0.95, min_df=10, max_features=5000,
        token_pattern=r"[a-z]{3,}",
        stop_words=STOP_WORDS_CUSTOM
    )
    matrice_comptes = vectoriseur.fit_transform(textes_nettoyes)
    print(f"Matrice : {matrice_comptes.shape[0]} docs × {matrice_comptes.shape[1]} mots")

    print(f"\nEntraînement LDA avec {n_themes} thèmes...")
    modele_lda = LatentDirichletAllocation(
        n_components=n_themes, random_state=42, max_iter=20, n_jobs=-1, verbose=1
    )
    modele_lda.fit(matrice_comptes)
    print("Entraînement terminé.")

    return modele_lda, vectoriseur, matrice_comptes


def afficher_themes(modele_lda, vectoriseur, n_mots=N_MOTS_PAR_THEME):
    """Affiche les top mots de chaque thème (étape d'interprétation)."""
    noms_mots = vectoriseur.get_feature_names_out()

    print("\n" + "=" * 70)
    print(f"THÈMES DÉCOUVERTS PAR LDA ({modele_lda.n_components} thèmes)")
    print("=" * 70)

    for i, theme in enumerate(modele_lda.components_):
        indices_top = theme.argsort()[:-n_mots - 1:-1]
        mots_top = [noms_mots[j] for j in indices_top]
        print(f"\nThème {i:>2} : {', '.join(mots_top)}")


def assigner_themes(modele_lda, vectoriseur, textes_nettoyes):
    """Assigne le thème dominant (argmax) à chaque réplique."""
    matrice = vectoriseur.transform(textes_nettoyes)
    distributions = modele_lda.transform(matrice)
    themes_dominants = distributions.argmax(axis=1)
    return pd.Series(themes_dominants, index=textes_nettoyes.index, name="theme_dominant")


def sauvegarder_modele(modele_lda, vectoriseur):
    """Sauvegarde le modèle LDA et le vectoriseur en .joblib."""
    joblib.dump(modele_lda, CHEMIN_MODELE_LDA)
    print(f"Modèle LDA sauvegardé : {CHEMIN_MODELE_LDA}")
    joblib.dump(vectoriseur, CHEMIN_VECTORISEUR)
    print(f"Vectoriseur sauvegardé : {CHEMIN_VECTORISEUR}")


def charger_modele_lda():
    """Charge le modèle LDA et le vectoriseur. Retourne (modele_lda, vectoriseur)."""
    modele_lda = joblib.load(CHEMIN_MODELE_LDA)
    vectoriseur = joblib.load(CHEMIN_VECTORISEUR)
    print("Modèle LDA et vectoriseur chargés.")
    return modele_lda, vectoriseur


if __name__ == "__main__":
    from src.preprocessing.load_cornell import charger_cornell

    print("=== MODÈLE B — EXTRACTION DE THÈMES PAR LDA ===\n")

    df_cornell = charger_cornell()
    df_analyse = df_cornell[df_cornell["character_gender"].isin(["m", "f"])].copy()
    print(f"\nRépliques avec genre connu : {len(df_analyse)}")

    textes_propres = nettoyer_repliques(df_analyse, col_texte="text")
    modele_lda, vectoriseur, matrice = entrainer_lda(textes_propres, n_themes=N_THEMES)
    afficher_themes(modele_lda, vectoriseur)

    themes = assigner_themes(modele_lda, vectoriseur, textes_propres)
    df_analyse.loc[textes_propres.index, "theme_dominant"] = themes

    print("\n--- Distribution des thèmes ---")
    print(df_analyse["theme_dominant"].value_counts().sort_index())

    sauvegarder_modele(modele_lda, vectoriseur)
    print("\n=== Modèle B terminé ===")
