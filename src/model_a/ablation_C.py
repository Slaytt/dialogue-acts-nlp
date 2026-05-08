"""
Ablation de l'hyperparamètre C de LinearSVC.
Entraîne le pipeline (TF-IDF + features) pour C ∈ {0.1, 1, 10},
puis évalue chaque variante et compile un tableau comparatif.
"""

import json
import os

from src.model_a.evaluate import evaluer
from src.model_a.train_classifier import entrainer

VALEURS_C = [0.1, 1.0, 10.0]
RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOSSIER_SORTIE = os.path.join(RACINE, "resultats", "model_a")


def main():
    resultats = []

    for C in VALEURS_C:
        nom = f"features_C{C}"
        print(f"\n{'='*70}\n  ABLATION C = {C}\n{'='*70}")
        entrainer(C=C, nom_run=nom)
        metrics = evaluer(nom_run=nom)
        resultats.append(metrics)

    # Récap
    print("\n" + "=" * 70)
    print("RÉCAP — Ablation C (avec features explicites)")
    print("=" * 70)
    print(f"{'C':<8} {'F1-macro':>10} {'F1-weighted':>12} {'accuracy':>10} {'F1 ORDRE':>10} {'F1 QUEST':>10} {'F1 DESACC':>10}")
    print("-" * 80)
    for r in resultats:
        C = r["run"].replace("features_C", "")
        print(
            f"{C:<8} "
            f"{r['f1_macro']:>10.3f} "
            f"{r['f1_weighted']:>12.3f} "
            f"{r['accuracy']:>10.3f} "
            f"{r['f1_par_classe']['ORDRE']:>10.3f} "
            f"{r['f1_par_classe']['QUESTION']:>10.3f} "
            f"{r['f1_par_classe']['DESACCORD']:>10.3f}"
        )

    # Référence baseline
    chemin_baseline = os.path.join(DOSSIER_SORTIE, "metrics_baseline.json")
    if os.path.exists(chemin_baseline):
        with open(chemin_baseline) as f:
            b = json.load(f)
        print(
            f"{'baseline':<8} "
            f"{b['f1_macro']:>10.3f} "
            f"{b['f1_weighted']:>12.3f} "
            f"{b['accuracy']:>10.3f} "
            f"{b['f1_par_classe']['ORDRE']:>10.3f} "
            f"{b['f1_par_classe']['QUESTION']:>10.3f} "
            f"{b['f1_par_classe']['DESACCORD']:>10.3f}"
        )


if __name__ == "__main__":
    main()
