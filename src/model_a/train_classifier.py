"""
Entraînement du Modèle A — TF-IDF + features explicites + LinearSVC.
Utilise le cache (cache_dataset.py) et les features supplémentaires (features.py).
"""

import os

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.preprocessing.cache_dataset import charger_dataset_clean
from src.preprocessing.features import NOMS_FEATURES, ajouter_features_au_df

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
    """Charge le cache, ajoute les features, retourne X (DataFrame), y."""
    df = ajouter_features_au_df(df)
    X = df[["texte_nettoye"] + NOMS_FEATURES]
    y = df["macro_classe"]
    return X, y


def entrainer(C=1.0, nom_run="features_v1"):
    print(f"=== Entraînement : {nom_run} (C={C}) ===\n")

    df = charger_dataset_clean()
    X, y = preparer_donnees(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train : {len(X_train)} | Test : {len(X_test)}")

    pipeline = construire_pipeline(C=C)
    print(f"Entraînement (C={C})...")
    pipeline.fit(X_train, y_train)

    print("\n--- Évaluation rapide sur le test set ---")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))

    chemin = os.path.join(RACINE, "src", "model_a", f"modele_{nom_run}.joblib")
    joblib.dump(pipeline, chemin)
    print(f"\nModèle sauvegardé : {chemin}")

    return pipeline


if __name__ == "__main__":
    entrainer(C=1.0, nom_run="features_v3_amp10")
