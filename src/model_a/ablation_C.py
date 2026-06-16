# Ablation de l'hyperparamètre C de LinearSVC.
# Protocole correct : on entraîne sur train (713 conv), on SCORE chaque C sur VAL (179 conv).
# Le test (223 conv) n'est touché qu'une seule fois, pour le C* retenu.

import json
import os

from src.model_a.evaluate import evaluer
from src.model_a.train_classifier import entrainer

VALEURS_C = [0.1, 1.0, 10.0]
RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOSSIER_SORTIE = os.path.join(RACINE, "resultats", "model_a")


def main():
    resultats_val = []

    for C in VALEURS_C:
        nom = f"features_C{C}"
        print(f"\n{'=' * 70}\n  ABLATION C = {C}  (sélection sur val)\n{'=' * 70}")
        # Entraîne sur train (713 conv), modèle sauvegardé. Affichage rapide val.
        entrainer(C=C, nom_run=nom, avec_val=True, evaluer_sur="val")
        # Métriques persistées sur VAL (et non test) pour la sélection.
        metrics = evaluer(nom_run=nom, sur="val", avec_val=True)
        metrics["C"] = C
        resultats_val.append(metrics)

    print("\n" + "=" * 70)
    print("RÉCAP ablation C — métriques sur VAL (179 conv)")
    print("=" * 70)
    print(f"{'C':<8} {'F1-macro':>10} {'F1-weighted':>12} {'accuracy':>10} {'F1 ORDRE':>10} {'F1 QUEST':>10} {'F1 DESACC':>10}")
    print("-" * 80)
    for r in resultats_val:
        print(
            f"{r['C']:<8} "
            f"{r['f1_macro']:>10.3f} "
            f"{r['f1_weighted']:>12.3f} "
            f"{r['accuracy']:>10.3f} "
            f"{r['f1_par_classe']['ORDRE']:>10.3f} "
            f"{r['f1_par_classe']['QUESTION']:>10.3f} "
            f"{r['f1_par_classe']['DESACCORD']:>10.3f}"
        )

    # Sélection du C optimal sur F1-macro VAL.
    meilleur = max(resultats_val, key=lambda r: r["f1_macro"])
    C_star = meilleur["C"]
    print(f"\n>>> C* sélectionné sur val : C = {C_star}  (F1-macro val = {meilleur['f1_macro']:.3f})")

    # Évaluation UNIQUE sur test pour C*.
    print(f"\n{'=' * 70}\n  ÉVALUATION FINALE sur TEST pour C* = {C_star}\n{'=' * 70}")
    metrics_test = evaluer(nom_run=f"features_C{C_star}", sur="test", avec_val=True)

    recap = {
        "valeurs_C_testees": VALEURS_C,
        "metriques_val": [{"C": r["C"], "f1_macro": r["f1_macro"]} for r in resultats_val],
        "C_star": C_star,
        "f1_macro_val_C_star": meilleur["f1_macro"],
        "f1_macro_test_C_star": metrics_test["f1_macro"],
    }
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    with open(os.path.join(DOSSIER_SORTIE, "ablation_C_recap.json"), "w") as f:
        json.dump(recap, f, indent=2, ensure_ascii=False)
    print(f"\nRécap : {DOSSIER_SORTIE}/ablation_C_recap.json")


if __name__ == "__main__":
    main()
