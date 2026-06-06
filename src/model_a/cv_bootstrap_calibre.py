import json
import os
import time

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.model_a.calibration import calibrer, predire_avec_betas
from src.model_a.train_classifier import construire_pipeline, preparer_donnees
from src.preprocessing.cache_dataset import charger_dataset_clean

RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHEMIN_SORTIE = os.path.join(RACINE, "resultats", "model_a", "cv_bootstrap_calibre.json")

C_RETENU = 0.1
N_FOLDS = 5
N_BOOTSTRAP = 1000
SEED = 42
TAILLE_VAL_INTERNE = 0.2  # 20% du train du fold → 16% du dataset total


def cv_calibree(X, y, classes):
    y_arr = np.asarray(y)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    y_pred_oof = np.empty(len(y_arr), dtype=object)
    f1_par_fold = []
    betas_par_fold = []

    for k, (idx_train_full, idx_test) in enumerate(cv.split(X, y_arr)):
        t0 = time.time()
        X_train_full = X.iloc[idx_train_full]
        y_train_full = y_arr[idx_train_full]
        X_test_fold = X.iloc[idx_test]
        y_test_fold = y_arr[idx_test]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=TAILLE_VAL_INTERNE, random_state=SEED, stratify=y_train_full
        )

        pipeline = construire_pipeline(C=C_RETENU)
        pipeline.fit(X_train, y_train)

        scores_val = pipeline.decision_function(X_val)
        betas = calibrer(scores_val, y_val, classes)
        betas_par_fold.append(betas)

        y_pred_fold = predire_avec_betas(pipeline, X_test_fold, classes, betas)
        y_pred_oof[idx_test] = y_pred_fold

        f1_fold = f1_score(y_test_fold, y_pred_fold, average="macro", zero_division=0)
        f1_par_fold.append(f1_fold)
        print(f"Fold {k+1}/{N_FOLDS} : F1-macro={f1_fold:.4f}  ({time.time()-t0:.1f}s)")

    return y_pred_oof, np.array(f1_par_fold), betas_par_fold


def bootstrap_f1(y_true, y_pred, n_iter, classes, rng):
    n = len(y_true)
    macro_scores = np.empty(n_iter)
    par_classe = {c: np.empty(n_iter) for c in classes}

    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        macro_scores[i] = f1_score(yt, yp, average="macro", zero_division=0)
        per_class = f1_score(yt, yp, labels=classes, average=None, zero_division=0)
        for c, s in zip(classes, per_class):
            par_classe[c][i] = s

    def ic(arr):
        return {
            "point": float(np.mean(arr)),
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
        }

    return ic(macro_scores), {c: ic(par_classe[c]) for c in classes}


def main():
    t0 = time.time()
    print(f"=== CV {N_FOLDS}-fold CALIBRÉ + Bootstrap (n_iter={N_BOOTSTRAP}) ===\n")

    df = charger_dataset_clean()
    X, y = preparer_donnees(df)
    y_arr = np.asarray(y)
    classes = sorted(np.unique(y_arr).tolist())
    print(f"Données : {len(y_arr)} exemples | {len(classes)} classes")
    print(f"Splits par fold : train 64% / val (calibration) 16% / test 20%\n")

    print("--- CV calibrée par fold ---")
    t1 = time.time()
    y_pred_oof, f1_folds, betas_par_fold = cv_calibree(X, y, classes)
    print(f"\nMoyenne ± std : {f1_folds.mean():.4f} ± {f1_folds.std():.4f}")
    print(f"(temps CV : {time.time()-t1:.1f}s)")

    f1_macro_oof = f1_score(y_arr, y_pred_oof, average="macro", zero_division=0)
    print(f"F1-macro out-of-fold (point estimate) : {f1_macro_oof:.4f}")

    print(f"\n--- Bootstrap {N_BOOTSTRAP} itérations sur prédictions OOF calibrées ---")
    t2 = time.time()
    rng = np.random.default_rng(SEED)
    macro_ci, par_classe_ci = bootstrap_f1(y_arr, y_pred_oof, N_BOOTSTRAP, classes, rng)
    print(f"(temps bootstrap : {time.time()-t2:.1f}s)")
    print(
        f"F1-macro IC 95% : [{macro_ci['ci_low']:.4f}, {macro_ci['ci_high']:.4f}] "
        f"(point {macro_ci['point']:.4f})"
    )

    print("\nIC par classe :")
    for c in classes:
        d = par_classe_ci[c]
        print(f"  {c:20s} {d['point']:.4f}  [{d['ci_low']:.4f}, {d['ci_high']:.4f}]")

    sortie = {
        "config": {
            "C": C_RETENU,
            "n_folds": N_FOLDS,
            "n_bootstrap": N_BOOTSTRAP,
            "seed": SEED,
            "calibration_par_fold": True,
            "splits": "train 64% / val 16% / test 20%",
        },
        "cv_f1_macro": {
            "scores": [float(s) for s in f1_folds],
            "mean": float(f1_folds.mean()),
            "std": float(f1_folds.std()),
        },
        "oof_f1_macro_point": float(f1_macro_oof),
        "bootstrap_f1_macro": macro_ci,
        "bootstrap_par_classe": par_classe_ci,
        "betas_par_fold": [{c: float(b) for c, b in d.items()} for d in betas_par_fold],
        "duree_totale_s": round(time.time() - t0, 1),
    }

    os.makedirs(os.path.dirname(CHEMIN_SORTIE), exist_ok=True)
    with open(CHEMIN_SORTIE, "w") as f:
        json.dump(sortie, f, indent=2, ensure_ascii=False)
    print(f"\nSauvegardé : {CHEMIN_SORTIE}")
    print(f"Durée totale : {sortie['duree_totale_s']}s")


if __name__ == "__main__":
    main()
