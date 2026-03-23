# Prétraitement des données SWDA
# nettoyer_texte()        → nettoie une réplique brute avec spaCy
# preparer_dataset_swda() → prépare le dataset complet pour l'entraînement

import os
import re

import pandas as pd
import spacy
from tqdm import tqdm

# Force le mode offline HuggingFace (utilise le cache local)
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Mapping 43 labels SWDA → 9 macro-classes (None = BRUIT, supprimé)
LABEL_TO_MACROCLASSE = {
    "sd": "STATEMENT",       # statement-non-opinion
    "+": "+",                # continuation — résolu par forward-fill dans preparer_dataset_swda()
    "sv": "OPINION",         # statement-opinion
    "bf": "OPINION",         # belief
    "qy": "QUESTION",        # yes-no question
    "qw": "QUESTION",        # wh-question
    "qh": "QUESTION",        # rhetorical question
    "qo": "QUESTION",        # open question
    "qrr": "QUESTION",       # or-clause
    "qy^d": "QUESTION",      # declarative yes-no
    "qw^d": "QUESTION",      # declarative wh
    "ad": "ORDRE",           # action-directive
    "aa": "ACCORD",          # accept
    "aap_am": "ACCORD",      # accept-part / maybe
    "ny": "ACCORD",          # yes-answer
    "no": "DESACCORD",       # no-answer
    "nn": "DESACCORD",       # no (neutre)
    "ng": "DESACCORD",       # disagree
    "ar": "DESACCORD",       # reject
    "arp_nd": "DESACCORD",   # reject-part
    "bd": "DESACCORD",       # downplayer / plainte
    "fa": "POLITESSE",       # apology
    "ft": "POLITESSE",       # thanking
    "fp": "POLITESSE",       # conventional-opening
    "fc": "POLITESSE",       # conventional-closing
    "b": "BACKCHANNEL",      # backchannel
    "b^m": "BACKCHANNEL",    # backchannel partiel
    "bh": "BACKCHANNEL",     # backchannel question
    "bk": "BACKCHANNEL",     # acknowledge-answer
    "ba": "BACKCHANNEL",     # assessment
    "br": "BACKCHANNEL",     # repeat
    "^2": "AUTRE_DIALOGUE",  # collaborative completion
    "^g": "AUTRE_DIALOGUE",  # tag-question
    "^h": "AUTRE_DIALOGUE",  # hold
    "^q": "AUTRE_DIALOGUE",  # citation
    "h": "AUTRE_DIALOGUE",   # hedge
    "%": None,               # fragment abandonné
    "x": None,               # non-verbal
    "t1": None,              # self-talk
    "t3": None,              # joke/anecdote (trop rare)
    "na": None,              # affirmation négative ambiguë
    'fo_o_fw_"_by_bc': None, # formules diverses
    "oo_co_cc": None,        # offre/option/accord conditionnel
}

# Stop words spaCy réhabilités car discriminants pour la classification
MOTS_A_GARDER = {
    "do", "does", "did",             # auxiliaires
    "can", "could", "would",         # modaux
    "will", "should", "might",
    "please",                        # ORDRE / POLITESSE
    "who", "what", "where",          # interrogatifs
    "when", "why", "how",
    "get", "make",                   # verbes d'action
    "now",                           # urgence
    "not", "no",                     # négation
    "say", "go", "take", "give",
    "put", "see", "keep", "call",
    "show", "move",
}


def nettoyer_texte(texte, nlp):
    """
    Nettoie une réplique brute → chaîne de lemmes utiles.
    Ex: "Get in the car right now!" → "get car right now"
    """
    # Suppression des marqueurs SWDA ({D ...}, [...], <<pause>>, <laughter>)
    texte = re.sub(r"\{[^}]*\}", "", texte)
    texte = re.sub(r"\[", "", texte)
    texte = re.sub(r"\]", "", texte)
    texte = re.sub(r"<<[^>]*>>", "", texte)
    texte = re.sub(r"<[^>]*>", "", texte)

    # Minuscule avant spaCy pour que MOTS_A_GARDER soient reconnus en début de phrase
    texte = texte.lower()

    doc = nlp(texte)

    lemmes_utiles = []
    for token in doc:
        lemme = token.lemma_.lower()
        # Garde : alphabétique ou dans MOTS_A_GARDER, pas stop word, len > 1
        # Le "or lemme in MOTS_A_GARDER" récupère "n't" → lemme "not"
        if (token.is_alpha or lemme in MOTS_A_GARDER) and not token.is_stop and len(lemme) > 1:
            lemmes_utiles.append(lemme)

    return " ".join(lemmes_utiles)


def preparer_dataset_swda(dataset):
    """
    Dataset SWDA brut (HuggingFace) → DataFrame prêt pour l'entraînement.
    Colonnes retournées : text, texte_nettoye, macro_classe
    """
    print("Chargement du modèle spaCy...")
    nlp_rapide = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    for mot in MOTS_A_GARDER:
        nlp_rapide.vocab[mot].is_stop = False
    nlp_rapide.vocab["n't"].is_stop = False

    print("Conversion du dataset en DataFrame...")
    df = pd.DataFrame(dataset["train"])

    # Labels entiers HuggingFace → noms textuels → macro-classes
    feature_label = dataset["train"].features["damsl_act_tag"]
    df["label_nom"] = df["damsl_act_tag"].apply(lambda i: feature_label.int2str(i))
    df["macro_classe"] = df["label_nom"].map(LABEL_TO_MACROCLASSE)

    # Résolution des continuations (+) : forward-fill PAR LOCUTEUR
    # (par conversation seule, un '+' de speaker A hériterait du backchannel de B)
    df = df.sort_values(
        ["conversation_no", "utterance_index", "subutterance_index"]
    ).reset_index(drop=True)
    df["macro_classe"] = df["macro_classe"].replace("+", pd.NA)
    df["macro_classe"] = df.groupby(["conversation_no", "caller"])["macro_classe"].ffill()

    # Suppression BRUIT
    nb_avant = len(df)
    df = df[df["macro_classe"].notna()].copy()
    print(
        f"Lignes supprimées (BRUIT) : {nb_avant - len(df)} "
        f"({(nb_avant - len(df)) / nb_avant * 100:.1f}%)"
    )

    # Nettoyage spaCy
    print("Nettoyage des textes avec spaCy...")
    tqdm.pandas(desc="Nettoyage spaCy")
    df["texte_nettoye"] = df["text"].progress_apply(
        lambda t: nettoyer_texte(str(t), nlp_rapide)
    )

    df = df[df["texte_nettoye"].str.strip() != ""].copy()

    print(
        f"\nDataset prêt : {len(df)} répliques | "
        f"{df['macro_classe'].nunique()} macro-classes"
    )
    print("\nDistribution des macro-classes :")
    print(df["macro_classe"].value_counts())

    return df[["text", "texte_nettoye", "macro_classe"]].reset_index(drop=True)


if __name__ == "__main__":
    from datasets import load_dataset

    print("=== TEST DU PRÉTRAITEMENT (30 premiers exemples) ===\n")
    dataset = load_dataset("swda", trust_remote_code=True)
    dataset_reduit = {"train": dataset["train"].select(range(30))}
    df_propre = preparer_dataset_swda(dataset_reduit)
    print("\n--- Résultat ---")
    print(df_propre.to_string())
