# Prédiction d'intentions — Modèle A
# IMPORTANT : le preprocessing DOIT être identique à celui de l'entraînement.

import os
import joblib
import spacy
import pandas as pd
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.preprocessing.clean_text import nettoyer_texte, MOTS_A_GARDER


def charger_modele(chemin=None):
    """Charge le pipeline (TF-IDF + LinearSVC) depuis le .joblib."""
    if chemin is None:
        chemin = os.path.join(os.path.dirname(__file__), "modele_dialogue_acts.joblib")

    print(f"Chargement du modèle depuis : {chemin}")
    pipeline = joblib.load(chemin)
    print("Modèle chargé avec succès.")
    return pipeline


def predire_intention(texte_brut, pipeline, nlp):
    """Prédit la macro-classe d'une réplique brute. Retourne "VIDE" si texte vide après nettoyage."""
    texte_nettoye = nettoyer_texte(str(texte_brut), nlp)

    if not texte_nettoye.strip():
        return "VIDE"

    # Le pipeline attend un DataFrame (ColumnTransformer avec 2 colonnes)
    contient_point = 1 if "?" in str(texte_brut) else 0
    df_input = pd.DataFrame({
        "texte_nettoye": [texte_nettoye],
        "contient_point_interrogation": [contient_point]
    })
    return pipeline.predict(df_input)[0]


def predire_sur_dataframe(df, col_texte, pipeline, nlp):
    """Applique la prédiction sur une colonne entière. Ajoute 'intention_predite'."""
    print(f"Prédiction sur {len(df)} répliques...")

    df = df.copy()
    df["intention_predite"] = df[col_texte].apply(
        lambda texte: predire_intention(texte, pipeline, nlp)
    )

    print("Prédiction terminée.")
    print("\nDistribution des intentions prédites :")
    print(df["intention_predite"].value_counts())
    return df


if __name__ == "__main__":
    pipeline = charger_modele()
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    # Réhabilitation des stop words (identique à l'entraînement)
    for mot in MOTS_A_GARDER:
        nlp.vocab[mot].is_stop = False
    nlp.vocab["n't"].is_stop = False

    exemples = [
        ("Get in the car right now!",           "ORDRE attendu"),
        ("Do you want some coffee?",             "QUESTION attendu"),
        ("I think this is a great idea.",        "OPINION attendu"),
        ("I'm so sorry for what happened.",      "POLITESSE attendu"),
        ("Yeah, absolutely.",                    "ACCORD attendu"),
        ("Uh-huh.",                              "BACKCHANNEL attendu"),
        ("That's not right at all.",             "DESACCORD attendu"),
        ("Life is complicated, you know.",       "STATEMENT attendu"),
        ("You never listen to me!",              "PLAINTE attendu"),
    ]

    print("\n" + "=" * 65)
    print(f"{'RÉPLIQUE':<40} {'ATTENDU':<20} {'PRÉDIT'}")
    print("=" * 65)

    for texte, attendu in exemples:
        predit = predire_intention(texte, pipeline, nlp)
        attendu_court = attendu.replace(" attendu", "")
        icone = "✓" if predit == attendu_court else "✗"
        print(f"{texte:<40} {attendu:<20} {predit} {icone}")
