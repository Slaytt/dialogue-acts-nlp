"""
Évaluation du Modèle A sur le test set SWDA.
Reproduit le split de train_classifier.py et persiste les métriques.
"""

import json
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from src.preprocessing.cache_dataset import charger_dataset_clean
from src.preprocessing.features import NOMS_FEATURES, ajouter_features_au_df

RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOSSIER_SORTIE = os.path.join(RACINE, "resultats", "model_a")


def reproduire_split(df):
    """Recrée X_test, y_test à l'identique de train_classifier.py."""
    df = ajouter_features_au_df(df)
    X = df[["texte_nettoye"] + NOMS_FEATURES]
    y = df["macro_classe"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test


def sauvegarder_matrice_confusion(y_true, y_pred, classes, chemin_png):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Vrai")
    ax.set_title("Matrice de confusion (normalisée par classe vraie)")

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j, i, f"{cm_norm[i, j]:.2f}",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontsize=8,
            )

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(chemin_png, dpi=150)
    plt.close()


def evaluer(nom_run="baseline", chemin_modele=None):
    print(f"=== Évaluation : {nom_run} ===\n")

    if chemin_modele is None:
        chemin_modele = os.path.join(RACINE, "src", "model_a", f"modele_{nom_run}.joblib")
        if not os.path.exists(chemin_modele):
            chemin_modele = os.path.join(RACINE, "src", "model_a", "modele_dialogue_acts.joblib")

    df = charger_dataset_clean()
    X_test, y_test = reproduire_split(df)

    print(f"Chargement du modèle : {chemin_modele}")
    pipeline = joblib.load(chemin_modele)

    print("Prédiction sur le test set...")
    y_pred = pipeline.predict(X_test)

    classes = sorted(y_test.unique())
    rapport = classification_report(y_test, y_pred, labels=classes, digits=3)
    print("\n" + rapport)

    metrics = {
        "run": nom_run,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "f1_par_classe": {
            c: f1_score(y_test, y_pred, labels=[c], average="macro")
            for c in classes
        },
        "support_test": {c: int((y_test == c).sum()) for c in classes},
    }

    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    with open(os.path.join(DOSSIER_SORTIE, f"metrics_{nom_run}.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(os.path.join(DOSSIER_SORTIE, f"classification_report_{nom_run}.txt"), "w") as f:
        f.write(rapport)
    sauvegarder_matrice_confusion(
        y_test, y_pred, classes,
        os.path.join(DOSSIER_SORTIE, f"confusion_matrix_{nom_run}.png"),
    )

    print(f"\nRésultats persistés dans : {DOSSIER_SORTIE}/")
    print(f"  - metrics_{nom_run}.json")
    print(f"  - classification_report_{nom_run}.txt")
    print(f"  - confusion_matrix_{nom_run}.png")

    return metrics


if __name__ == "__main__":
    nom = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    evaluer(nom_run=nom)
