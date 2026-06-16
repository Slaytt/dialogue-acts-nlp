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


# --- Features POS (mode impératif) -----------------------------------------
# Calculées via le tagger spaCy (pas le parser) sur le texte CASÉ, après strip
# des marqueurs SWDA. Conçues à partir de l'analyse d'erreurs ORDRE (2026-06-08) :
#   feat_imperatif_2p : impératif 2e personne (verbe base en tête, pas de sujet)
#                       → généralise les impératifs nus que feat_verbe_action ratait.
#   feat_hortatif_1p  : hortatif 1re personne (let me / let's) → isole le motif qui
#                       plombait la précision (le SVM lui donne sa propre coordonnée).

# Marqueurs de discours sautés en tête pour atteindre le vrai début de l'énoncé.
# N'inclut PAS les pronoms sujets (leur présence signale justement un non-impératif).
MARQUEURS_DISCOURS = {
    "well", "now", "so", "just", "please", "actually", "anyway", "like",
    "see", "oh", "yeah", "okay", "ok", "um", "uh", "hey", "right", "and",
    "but", "or",
}
PRONOMS_SUJET = {"you", "we", "they", "i", "he", "she", "it"}


def classifie_imperatif(doc):
    """(feat_imperatif_2p, feat_hortatif_1p) pour un Doc spaCy taggé."""
    toks = [t for t in doc if not t.is_space and t.text.strip()]
    i = 0
    while i < len(toks) and (
        toks[i].pos_ in {"PUNCT", "CCONJ", "INTJ", "SPACE", "SYM"}
        or toks[i].lower_ in MARQUEURS_DISCOURS
    ):
        i += 1
    if i >= len(toks):
        return 0, 0
    head = toks[i]
    nxt = toks[i + 1] if i + 1 < len(toks) else None

    # Hortatif 1re personne : let me / let's / let us / lets
    if head.lower_ == "lets":
        return 0, 1
    if head.lower_ == "let" and nxt is not None and nxt.lower_ in {"me", "us", "'s"}:
        return 0, 1

    # Question avec do-support ("do you ...", "does he ...") → pas un impératif
    if head.lemma_ == "do" and nxt is not None and nxt.lower_ in PRONOMS_SUJET:
        return 0, 0

    # Impératif : 1er token de contenu = verbe forme base (VB) → aucun sujet devant
    if head.tag_ == "VB":
        return 1, 0
    return 0, 0


def _charger_tagger():
    import spacy
    if not hasattr(_charger_tagger, "_nlp"):
        _charger_tagger._nlp = spacy.load(
            "en_core_web_sm", disable=["parser", "ner", "lemmatizer"]
        )
    return _charger_tagger._nlp


def ajouter_features_pos(df, nlp=None):
    """Ajoute feat_imperatif_2p / feat_hortatif_1p via le tagger spaCy.
    Coûteux (un passage tagger par énoncé) → à précalculer dans le cache."""
    df = df.copy()
    if nlp is None:
        nlp = _charger_tagger()
    textes = [nettoyer_marqueurs_swda(str(t)) for t in df["text"]]
    imp, hort = [], []
    for d in nlp.pipe(textes, batch_size=256):
        a, b = classifie_imperatif(d)
        imp.append(a)
        hort.append(b)
    df["feat_imperatif_2p"] = imp
    df["feat_hortatif_1p"] = hort
    return df


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
    "feat_imperatif_2p",
    "feat_hortatif_1p",
]


def ajouter_features_au_df(df):
    df = df.copy()
    df["contient_point_interrogation"] = df["text"].apply(
        lambda x: 1 if "?" in str(x) else 0
    )
    feats = df["text"].apply(extraire_features).apply(pd.Series)
    df = pd.concat([df, feats], axis=1)
    # Features POS : viennent du cache si présentes, sinon calculées à la volée.
    if "feat_imperatif_2p" not in df.columns or "feat_hortatif_1p" not in df.columns:
        df = ajouter_features_pos(df)
    return df


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
