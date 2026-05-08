# Calibration de seuil par coordinate descent.
# Sur le pipeline TF-IDF + features + LinearSVC, ajoute un biais beta_c par classe
# au decision_function avant l'argmax, optimisé pour maximiser F1-macro sur un set de validation.

import json
import os

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split

from src.model_a.evaluate import sauvegarder_matrice_confusion
from src.model_a.train_classifier import construire_pipeline
from src.preprocessing.cache_dataset import charger_dataset_clean
from src.preprocessing.features import NOMS_FEATURES, ajouter_features_au_df

RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOSSIER_SORTIE = os.path.join(RACINE, "resultats", "model_a")


def split_train_val_test(df, noms_features):
    df = ajouter_features_au_df(df)
    X = df[["texte_nettoye"] + noms_features]
    y = df["macro_classe"]

    X_tr_full, X_test, y_tr_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr_full, y_tr_full, test_size=0.2, random_state=42, stratify=y_tr_full
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def calibrer(scores_val, y_val, classes, n_passes=3, pas=0.05, beta_min=-1.0, beta_max=1.0):
    classe_idx = {c: i for i, c in enumerate(classes)}
    betas = {c: 0.0 for c in classes}
    grille_beta = np.arange(beta_min, beta_max + pas / 2, pas)

    def predire(b_dict):
        scores_mod = scores_val.copy()
        for c, b in b_dict.items():
            scores_mod[:, classe_idx[c]] += b
        idx_pred = scores_mod.argmax(axis=1)
        return np.array([classes[j] for j in idx_pred])

    f1_actuel = f1_score(y_val, predire(betas), average="macro")
    print(f"F1-macro initial (sans calibration) : {f1_actuel:.4f}")

    for passe in range(n_passes):
        for c in classes:
            meilleur_beta = betas[c]
            meilleur_f1 = f1_actuel
            for beta in grille_beta:
                test_betas = {**betas, c: float(beta)}
                f1 = f1_score(y_val, predire(test_betas), average="macro")
                if f1 > meilleur_f1:
                    meilleur_f1 = f1
                    meilleur_beta = float(beta)
            betas[c] = meilleur_beta
            f1_actuel = meilleur_f1
        print(f"Passe {passe + 1}/{n_passes} : F1-macro = {f1_actuel:.4f}")

    return betas


def predire_avec_betas(pipeline, X, classes, betas):
    scores = pipeline.decision_function(X)
    classe_idx = {c: i for i, c in enumerate(classes)}
    scores_mod = scores.copy()
    for c, b in betas.items():
        scores_mod[:, classe_idx[c]] += b
    idx_pred = scores_mod.argmax(axis=1)
    return np.array([classes[j] for j in idx_pred])


def main(C=0.1, nom_run="features_C0.1_calibre"):
    print(f"=== Calibration : {nom_run} (C={C}) ===\n")

    df = charger_dataset_clean()
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(df, NOMS_FEATURES)
    classes = sorted(y_train.unique())
    print(f"Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}\n")

    print(f"Entraînement du pipeline (C={C})...")
    pipeline = construire_pipeline(C=C, noms_features=NOMS_FEATURES)
    pipeline.fit(X_train, y_train)

    print("Calcul des scores sur val...")
    scores_val = pipeline.decision_function(X_val)

    print("\n--- Coordinate descent ---")
    betas = calibrer(scores_val, y_val.values, classes)

    betas_non_nuls = {c: b for c, b in betas.items() if abs(b) > 0.001}
    print(f"\nBetas appris (non nuls) :")
    for c, b in sorted(betas_non_nuls.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {c:<18} {b:+.3f}")

    print("\n--- Évaluation sur test set ---")
    y_pred = predire_avec_betas(pipeline, X_test, classes, betas)
    rapport = classification_report(y_test, y_pred, labels=classes, digits=3)
    print(rapport)

    metrics = {
        "run": nom_run,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "f1_par_classe": {
            c: f1_score(y_test, y_pred, labels=[c], average="macro") for c in classes
        },
        "support_test": {c: int((y_test == c).sum()) for c in classes},
        "betas": betas,
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

    chemin_modele = os.path.join(RACINE, "src", "model_a", f"modele_{nom_run}.joblib")
    joblib.dump({"pipeline": pipeline, "betas": betas, "classes": classes}, chemin_modele)
    print(f"\nModèle calibré sauvegardé : {chemin_modele}")

    return metrics


if __name__ == "__main__":
    main()
