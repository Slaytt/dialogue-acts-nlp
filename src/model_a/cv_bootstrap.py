import json
import os
import time

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

from src.model_a.train_classifier import construire_pipeline, preparer_donnees
from src.preprocessing.cache_dataset import charger_dataset_clean

RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHEMIN_SORTIE = os.path.join(RACINE, "resultats", "model_a", "cv_bootstrap.json")

C_RETENU = 0.1
N_FOLDS = 5
N_BOOTSTRAP = 1000
SEED = 42


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
    print(f"=== CV {N_FOLDS}-fold + Bootstrap (n_iter={N_BOOTSTRAP}) ===\n")

    df = charger_dataset_clean()
    X, y = preparer_donnees(df)
    y_arr = np.asarray(y)
    classes = sorted(np.unique(y_arr).tolist())
    print(f"Données : {len(y_arr)} exemples | {len(classes)} classes")

    pipeline = construire_pipeline(C=C_RETENU)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    print(f"\n--- cross_val_score (F1-macro) ---")
    t1 = time.time()
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"Scores par fold : {[round(s, 4) for s in cv_scores]}")
    print(f"Moyenne ± std : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"(temps CV : {time.time() - t1:.1f}s)")

    print(f"\n--- cross_val_predict (out-of-fold) ---")
    t2 = time.time()
    y_pred_oof = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=-1)
    print(f"(temps CV-predict : {time.time() - t2:.1f}s)")

    f1_macro_oof = f1_score(y_arr, y_pred_oof, average="macro", zero_division=0)
    print(f"F1-macro out-of-fold (point estimate) : {f1_macro_oof:.4f}")

    print(f"\n--- Bootstrap {N_BOOTSTRAP} itérations sur prédictions OOF ---")
    t3 = time.time()
    rng = np.random.default_rng(SEED)
    macro_ci, par_classe_ci = bootstrap_f1(
        y_arr, y_pred_oof, N_BOOTSTRAP, classes, rng
    )
    print(f"(temps bootstrap : {time.time() - t3:.1f}s)")
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
        },
        "cv_f1_macro": {
            "scores": [float(s) for s in cv_scores],
            "mean": float(cv_scores.mean()),
            "std": float(cv_scores.std()),
        },
        "oof_f1_macro_point": float(f1_macro_oof),
        "bootstrap_f1_macro": macro_ci,
        "bootstrap_par_classe": par_classe_ci,
        "duree_totale_s": round(time.time() - t0, 1),
    }

    os.makedirs(os.path.dirname(CHEMIN_SORTIE), exist_ok=True)
    with open(CHEMIN_SORTIE, "w") as f:
        json.dump(sortie, f, indent=2, ensure_ascii=False)
    print(f"\nSauvegardé : {CHEMIN_SORTIE}")
    print(f"Durée totale : {sortie['duree_totale_s']}s")


if __name__ == "__main__":
    main()
