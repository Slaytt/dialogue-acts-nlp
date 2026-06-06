"""Chargement de MapTask (instruct uniquement) depuis SILICONE → DataFrame compatible SWDA.

Stratégie d'augmentation pour la classe ORDRE (cf. Bloc 1bis du cadrage) :
- Source : eusip/silicone, config 'maptask', split 'train' seulement
- Filtre : Dialogue_Act == 'instruct'
- Mapping : instruct → ORDRE (option a, mapping unique défendable)
- Preprocessing : même nettoyer_texte que SWDA pour cohérence des features
"""

import os
import pandas as pd
import spacy
from datasets import load_dataset
from tqdm import tqdm

from src.preprocessing.clean_text import MOTS_A_GARDER, nettoyer_texte

DOSSIER_PROCESSED = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "processed"
)
CACHE_PATH = os.path.join(DOSSIER_PROCESSED, "maptask_clean.pkl")


def charger_maptask(split="train", use_cache=True):
    """Charge MapTask filtré sur 'instruct' → DataFrame [text, texte_nettoye, macro_classe].

    split: split SILICONE à utiliser (par défaut 'train' uniquement, cf. discussion
           hygiène scientifique : on ne touche pas au test set d'un benchmark public).
    use_cache: si True et qu'un pickle existe, charge le cache au lieu de rejouer spaCy.
    """
    if use_cache and split == "train" and os.path.exists(CACHE_PATH):
        print(f"Chargement depuis le cache : {CACHE_PATH}")
        return pd.read_pickle(CACHE_PATH)

    print(f"Téléchargement de eusip/silicone config=maptask split={split}...")
    ds = load_dataset("eusip/silicone", "maptask", trust_remote_code=True)
    df = pd.DataFrame(ds[split])

    nb_total = len(df)
    df = df[df["Dialogue_Act"] == "instruct"].copy()
    print(f"Filtrage Dialogue_Act=='instruct' : {len(df)} / {nb_total} utterances retenues")

    df = df.rename(columns={"Utterance": "text"})
    df["macro_classe"] = "ORDRE"

    print("Chargement du modèle spaCy...")
    nlp_rapide = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    for mot in MOTS_A_GARDER:
        nlp_rapide.vocab[mot].is_stop = False
    nlp_rapide.vocab["n't"].is_stop = False

    print("Nettoyage des textes avec spaCy...")
    tqdm.pandas(desc="Nettoyage spaCy")
    df["texte_nettoye"] = df["text"].progress_apply(
        lambda t: nettoyer_texte(str(t), nlp_rapide)
    )

    df = df[df["texte_nettoye"].str.strip() != ""].copy()
    df = df[["text", "texte_nettoye", "macro_classe"]].reset_index(drop=True)

    if split == "train":
        os.makedirs(DOSSIER_PROCESSED, exist_ok=True)
        df.to_pickle(CACHE_PATH)
        print(f"Cache écrit : {CACHE_PATH}")

    return df


def afficher_stats(df):
    print("\n" + "=" * 60)
    print("STATISTIQUES — MAPTASK 'instruct' → ORDRE")
    print("=" * 60)
    print(f"\nNombre d'utterances : {len(df):,}")
    print(f"Longueur moyenne (texte brut) : {df['text'].str.len().mean():.1f} caractères")
    print(f"Longueur moyenne (texte nettoyé) : {df['texte_nettoye'].str.len().mean():.1f} caractères")
    print(f"Utterances avec texte nettoyé vide (post-filtre) : 0 (déjà éliminées)")

    print("\n--- Aperçu (10 premières utterances) ---")
    for _, row in df.head(10).iterrows():
        print(f"  [{row['macro_classe']}] {row['text']!r}")
        print(f"      → nettoyé: {row['texte_nettoye']!r}")


if __name__ == "__main__":
    print("=== CHARGEMENT DE MAPTASK (instruct → ORDRE) ===\n")
    df = charger_maptask(split="train", use_cache=False)
    afficher_stats(df)
