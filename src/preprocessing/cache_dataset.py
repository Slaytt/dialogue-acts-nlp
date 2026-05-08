# src/preprocessing/cache_dataset.py
"""Cache du dataset SWDA nettoyé.
Premier run : ~5-10 min (téléchargement + nettoyage spaCy).
Runs suivants : ~1 sec (lecture du pickle).
"""

import os

import pandas as pd
from datasets import load_dataset

from src.preprocessing.clean_text import preparer_dataset_swda

# Chemin absolu vers data/processed/swda_clean.pkl
RACINE_PROJET = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CHEMIN_CACHE = os.path.join(RACINE_PROJET, "data", "processed", "swda_clean.pkl")


def charger_dataset_clean(forcer_recalcul=False):
    """
    Retourne le DataFrame SWDA nettoyé (text, texte_nettoye, macro_classe).

    Premier appel : exécute le pipeline complet et sauvegarde le résultat.
    Appels suivants : recharge depuis le pickle.

    forcer_recalcul=True : ignore le cache et reconstruit (utile si clean_text.py change).
    """
    if os.path.exists(CHEMIN_CACHE) and not forcer_recalcul:
        print(f"Chargement depuis le cache : {CHEMIN_CACHE}")
        df = pd.read_pickle(CHEMIN_CACHE)
        print(f"Cache chargé : {len(df)} répliques")
        return df

    print("Cache absent — construction du dataset (peut prendre 5-10 min)...")
    dataset = load_dataset("swda", trust_remote_code=True)
    df = preparer_dataset_swda(dataset)

    os.makedirs(os.path.dirname(CHEMIN_CACHE), exist_ok=True)
    df.to_pickle(CHEMIN_CACHE)
    print(f"Cache sauvegardé : {CHEMIN_CACHE}")

    return df


if __name__ == "__main__":
    df = charger_dataset_clean()
    print("\nAperçu :")
    print(df.head())
    print(f"\nTaille : {len(df)} | Classes : {df['macro_classe'].nunique()}")
