# Features booléennes ciblant ORDRE, QUESTION, DESACCORD, POLITESSE.
# Calculées par regex sur le texte brut (post nettoyage des marqueurs SWDA).

import re

import pandas as pd

VERBES_ACTION = {
    "get", "go", "take", "give", "stop", "come", "look", "listen",
    "sit", "stand", "wait", "tell", "put", "let", "show", "open",
    "close", "turn", "move", "hold", "leave", "try", "keep", "say",
    "make", "do", "don't",
}
INTERROGATIFS = {
    "what", "who", "where", "when", "why", "how",
    "do", "does", "did", "is", "are", "was", "were",
    "can", "could", "would", "will", "shall", "should", "have", "has",
}
NEGATIONS_DEBUT = {"no", "not", "nope", "nah", "never"}


def nettoyer_marqueurs_swda(texte):
    texte = re.sub(r"\{[^}]*\}", "", texte)
    texte = re.sub(r"[\[\]]", "", texte)
    texte = re.sub(r"<<[^>]*>>", "", texte)
    texte = re.sub(r"<[^>]*>", "", texte)
    texte = re.sub(r"/\s*$", "", texte)
    return texte.strip()


def extraire_features(texte_brut):
    texte = nettoyer_marqueurs_swda(str(texte_brut))
    texte_lower = texte.lower()

    mots = re.findall(r"[\w']+", texte_lower)
    premier = mots[0] if mots else ""

    return {
        "feat_exclamation":     int("!" in texte),
        "feat_now":             int(re.search(r"\bnow\b", texte_lower) is not None),
        "feat_verbe_action":    int(premier in VERBES_ACTION),
        "feat_modal_poli":      int(re.search(r"\b(could|would|will)\s+you\b", texte_lower) is not None),
        "feat_interrogatif":    int(premier in INTERROGATIFS),
        "feat_negation_debut":  int(premier in NEGATIONS_DEBUT),
        "feat_negation":        int(re.search(r"n't|\bnot\b|\bnever\b|\bno\b", texte_lower) is not None),
        "feat_mais":            int(re.search(r"\b(but|however|actually)\b", texte_lower) is not None),
        "feat_politesse":       int(re.search(r"\b(sorry|thank|please|excuse)\b", texte_lower) is not None),
    }


# Ordre fixe pour rester cohérent avec le ColumnTransformer
NOMS_FEATURES = [
    "contient_point_interrogation",
    "feat_exclamation",
    "feat_now",
    "feat_verbe_action",
    "feat_modal_poli",
    "feat_interrogatif",
    "feat_negation_debut",
    "feat_negation",
    "feat_mais",
    "feat_politesse",
]


def ajouter_features_au_df(df):
    df = df.copy()
    df["contient_point_interrogation"] = df["text"].apply(
        lambda x: 1 if "?" in str(x) else 0
    )
    feats = df["text"].apply(extraire_features).apply(pd.Series)
    return pd.concat([df, feats], axis=1)


if __name__ == "__main__":
    exemples = [
        "Get out of here!",
        "Could you pass the salt?",
        "I think this is wrong.",
        "No, that's not true.",
        "Thank you so much.",
        "Yeah right.",
        "What time is it?",
        "{F Uh, } I don't know.",
    ]
    print(f"{'Texte':<35} | features actives")
    print("-" * 80)
    for ex in exemples:
        feats = extraire_features(ex)
        actives = [k for k, v in feats.items() if v == 1]
        print(f"{ex[:35]:<35} | {', '.join(actives) if actives else '(aucune)'}")
