# Entraînement du Modèle A : TF-IDF (1-2 grammes) + features explicites + LinearSVC.

import os

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.preprocessing.cache_dataset import charger_dataset_clean
from src.preprocessing.features import NOMS_FEATURES, ajouter_features_au_df
from src.preprocessing.splits import split_par_conversation

RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def construire_pipeline(C=1.0, noms_features=None):
    if noms_features is None:
        noms_features = NOMS_FEATURES
    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf",
             TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True),
             "texte_nettoye"),
            ("features_supp", "passthrough", noms_features),
        ]
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LinearSVC(class_weight="balanced", max_iter=5000, C=C, random_state=42)),
    ])


def preparer_donnees(df):
    df = ajouter_features_au_df(df)
    X = df[["texte_nettoye"] + NOMS_FEATURES]
    y = df["macro_classe"]
    return X, y


def obtenir_splits(df, avec_val=True):
    """Wrapper unique : split par conversation, 64/16/20 si avec_val, 80/20 sinon.
    Garde conversation_no côté df mais l'exclut de X (features pipeline)."""
    df_feat = ajouter_features_au_df(df)
    if avec_val:
        df_tr, df_va, df_te = split_par_conversation(df_feat, val_size=0.2)
        splits = {"train": df_tr, "val": df_va, "test": df_te}
    else:
        df_tr, df_te = split_par_conversation(df_feat)
        splits = {"train": df_tr, "test": df_te}

    out = {}
    for nom, d in splits.items():
        out[f"X_{nom}"] = d[["texte_nettoye"] + NOMS_FEATURES]
        out[f"y_{nom}"] = d["macro_classe"]
    return out


def entrainer(C=1.0, nom_run="features", avec_val=True, evaluer_sur="val"):
    """
    avec_val=True  → split 64/16/20, modèle entraîné sur les 64% (713 conv)
    avec_val=False → split 80/20,   modèle entraîné sur les 80% (892 conv)
    evaluer_sur ∈ {"val","test"} — où l'évaluation rapide d'affichage est faite.
    Le test set n'est touché que pour le modèle final retenu.
    """
    print(f"=== Entraînement : {nom_run} (C={C}, avec_val={avec_val}) ===\n")

    df = charger_dataset_clean()
    s = obtenir_splits(df, avec_val=avec_val)
    if f"X_{evaluer_sur}" not in s:
        raise ValueError(f"evaluer_sur='{evaluer_sur}' indisponible (splits={list(s.keys())})")

    X_train, y_train = s["X_train"], s["y_train"]
    X_eval, y_eval = s[f"X_{evaluer_sur}"], s[f"y_{evaluer_sur}"]
    print(f"Train : {len(X_train)} | {evaluer_sur} : {len(X_eval)}")

    pipeline = construire_pipeline(C=C)
    print(f"Entraînement (C={C})...")
    pipeline.fit(X_train, y_train)

    print(f"\n--- Évaluation rapide sur {evaluer_sur} ---")
    y_pred = pipeline.predict(X_eval)
    print(classification_report(y_eval, y_pred, digits=3))

    chemin = os.path.join(RACINE, "src", "model_a", f"modele_{nom_run}.joblib")
    joblib.dump(pipeline, chemin)
    print(f"\nModèle sauvegardé : {chemin}")

    return pipeline


if __name__ == "__main__":
    entrainer(avec_val=True, evaluer_sur="val")
