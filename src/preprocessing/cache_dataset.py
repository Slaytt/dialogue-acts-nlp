# Cache pickle du DataFrame SWDA nettoyé.
# Évite de relancer spaCy (~5 min) à chaque expérimentation.

import os

import pandas as pd
from datasets import load_dataset

from src.preprocessing.clean_text import preparer_dataset_swda
from src.preprocessing.features import ajouter_features_pos

RACINE_PROJET = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CHEMIN_CACHE = os.path.join(RACINE_PROJET, "data", "processed", "swda_clean.pkl")

# Colonnes de features POS précalculées et stockées dans le pickle (cf. features.py).
COLS_FEATURES_POS = ["feat_imperatif_2p", "feat_hortatif_1p"]


def charger_dataset_clean(forcer_recalcul=False):
    if os.path.exists(CHEMIN_CACHE) and not forcer_recalcul:
        print(f"Chargement depuis le cache : {CHEMIN_CACHE}")
        df = pd.read_pickle(CHEMIN_CACHE)
        print(f"Cache chargé : {len(df)} répliques")
        return df

    print("Cache absent — construction du dataset (5-10 min)...")
    dataset = load_dataset("swda", trust_remote_code=True)
    df = preparer_dataset_swda(dataset)
    print("Calcul des features POS (tagger spaCy)...")
    df = ajouter_features_pos(df)

    os.makedirs(os.path.dirname(CHEMIN_CACHE), exist_ok=True)
    df.to_pickle(CHEMIN_CACHE)
    print(f"Cache sauvegardé : {CHEMIN_CACHE}")

    return df


def augmenter_cache_features_pos():
    """Ajoute les colonnes POS à un cache existant SANS rebuild complet.
    N'altère ni les lignes, ni leur ordre, ni conversation_no → le split
    (seed=42) reste strictement identique, test 20% intouché."""
    if not os.path.exists(CHEMIN_CACHE):
        raise FileNotFoundError("Cache absent — lance charger_dataset_clean() d'abord.")
    df = pd.read_pickle(CHEMIN_CACHE)
    print(f"Cache chargé : {len(df)} répliques")
    if all(c in df.columns for c in COLS_FEATURES_POS):
        print("Features POS déjà présentes — rien à faire.")
        return df
    print("Calcul des features POS (tagger spaCy, ~quelques min)...")
    df = ajouter_features_pos(df)
    df.to_pickle(CHEMIN_CACHE)
    print(f"Cache augmenté et resauvegardé : {CHEMIN_CACHE}")
    print(df[COLS_FEATURES_POS].sum().to_string())
    return df


if __name__ == "__main__":
    df = charger_dataset_clean()
    print("\nAperçu :")
    print(df.head())
    print(f"\nTaille : {len(df)} | Classes : {df['macro_classe'].nunique()}")
