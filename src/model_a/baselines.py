"""
Baselines pour le Modèle A.
Trois baselines : Dummy(most_frequent), Dummy(stratified), LogisticRegression.
Chaque baseline est entraînée sur les mêmes 80% que LinearSVC, évaluée sur les mêmes 20%.
"""

import json
import os

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing.cache_dataset import charger_dataset_clean

RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOSSIER_SORTIE = os.path.join(RACINE, "resultats", "model_a")


def preparer_split(df):
    """Split 80/20 stratifié, identique à train_classifier.py et evaluate.py."""
    df = df.copy()
    df["contient_point_interrogation"] = df["text"].apply(
        lambda x: 1 if "?" in str(x) else 0
    )
    X = df[["texte_nettoye", "contient_point_interrogation"]]
    y = df["macro_classe"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def construire_preprocesseur():
    """Même preprocessing que LinearSVC : TF-IDF (1-2 grammes, 50k feats) + ?."""
    return ColumnTransformer(
        transformers=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True),
             "texte_nettoye"),
            ("features_supp", "passthrough", ["contient_point_interrogation"]),
        ]
    )


def evaluer_modele(nom, modele, X_train, X_test, y_train, y_test, classes):
    """Entraîne, prédit, calcule + persiste les métriques pour un modèle."""
    print(f"\n--- Baseline : {nom} ---")
    modele.fit(X_train, y_train)
    y_pred = modele.predict(X_test)

    rapport = classification_report(y_test, y_pred, labels=classes, digits=3, zero_division=0)
    print(rapport)

    metrics = {
        "run": nom,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_par_classe": {
            c: f1_score(y_test, y_pred, labels=[c], average="macro", zero_division=0)
            for c in classes
        },
    }

    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    with open(os.path.join(DOSSIER_SORTIE, f"metrics_{nom}.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(os.path.join(DOSSIER_SORTIE, f"classification_report_{nom}.txt"), "w") as f:
        f.write(rapport)

    return metrics


def main():
    print("Chargement du dataset...")
    df = charger_dataset_clean()
    X_train, X_test, y_train, y_test = preparer_split(df)
    classes = sorted(y_train.unique())

    print(f"Train : {len(X_train)} | Test : {len(X_test)}\n")

    resultats = []

    # Baseline 1 — Dummy most_frequent
    pipe = Pipeline([
        ("preprocessor", construire_preprocesseur()),
        ("clf", DummyClassifier(strategy="most_frequent")),
    ])
    resultats.append(evaluer_modele(
        "dummy_most_frequent", pipe, X_train, X_test, y_train, y_test, classes
    ))

    # Baseline 2 — Dummy stratified
    pipe = Pipeline([
        ("preprocessor", construire_preprocesseur()),
        ("clf", DummyClassifier(strategy="stratified", random_state=42)),
    ])
    resultats.append(evaluer_modele(
        "dummy_stratified", pipe, X_train, X_test, y_train, y_test, classes
    ))

    # Baseline 3 — LogisticRegression
    pipe = Pipeline([
        ("preprocessor", construire_preprocesseur()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, n_jobs=-1)),
    ])
    resultats.append(evaluer_modele(
        "logreg_balanced", pipe, X_train, X_test, y_train, y_test, classes
    ))

    # Tableau récapitulatif
    print("\n" + "=" * 70)
    print("RÉCAP — Comparaison baselines vs LinearSVC (à charger séparément)")
    print("=" * 70)
    print(f"{'modèle':<25} {'accuracy':>10} {'F1-macro':>10} {'F1-weighted':>12}")
    print("-" * 70)
    for r in resultats:
        print(f"{r['run']:<25} {r['accuracy']:>10.3f} {r['f1_macro']:>10.3f} {r['f1_weighted']:>12.3f}")

    # Charger la baseline LinearSVC déjà calculée
    chemin_svc = os.path.join(DOSSIER_SORTIE, "metrics_baseline.json")
    if os.path.exists(chemin_svc):
        with open(chemin_svc) as f:
            svc = json.load(f)
        print(f"{'linearsvc (référence)':<25} {svc['accuracy']:>10.3f} {svc['f1_macro']:>10.3f} {svc['f1_weighted']:>12.3f}")

    print("\nMétriques persistées dans resultats/model_a/")


if __name__ == "__main__":
    main()
