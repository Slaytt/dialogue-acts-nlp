# Analyses avancées depuis resultats_complets.csv
# Lit le CSV déjà produit par merge_results.py — pas besoin de relancer les modèles.

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOSSIER_RESULTATS = os.path.join(os.path.dirname(__file__), "..", "..", "resultats")
CHEMIN_CSV = os.path.join(DOSSIER_RESULTATS, "resultats_complets.csv")

NOMS_THEMES = {
    0: "Amour / Croyances",
    1: "Foyer / Excuses",
    2: "Travail / Société",
    3: "Famille / Vie",
    4: "Quotidien / Mémoire",
    5: "Réflexion / Dialogue",
    6: "Action / Mouvement",
    7: "Espace domestique",
    8: "Ordres / Émotions",
    9: "Argent / Morale",
    10: "Violence / Confrontation",
    11: "Apparences / Rencontres",
}


def nommer_theme(num):
    return NOMS_THEMES.get(num, f"Thème {num}")


def charger_resultats():
    print(f"Chargement de {CHEMIN_CSV}...")
    df = pd.read_csv(CHEMIN_CSV)
    df["nom_theme"] = df["theme_dominant"].apply(nommer_theme)
    print(f"  → {len(df)} répliques | {df['theme_dominant'].nunique()} thèmes | "
          f"{df['intention_predite'].nunique()} intentions")
    return df


def analyser_intention_theme_genre(df):
    """Pour chaque thème, profil d'intentions H vs F (normalize=columns)."""
    print("\n" + "=" * 60)
    print("ANALYSE 1 : Intentions × Thèmes × Genre")
    print("=" * 60)

    resultats = {}
    for num_theme in sorted(df["theme_dominant"].unique()):
        nom = nommer_theme(num_theme)
        sous_df = df[df["theme_dominant"] == num_theme]

        tab = pd.crosstab(
            sous_df["intention_predite"], sous_df["character_gender"],
            normalize="columns"
        ) * 100
        tab = tab.rename(columns={"m": "Hommes (%)", "f": "Femmes (%)"})
        tab["Écart (F−H)"] = tab["Femmes (%)"] - tab["Hommes (%)"]
        tab = tab.sort_values("Écart (F−H)", key=abs, ascending=False)

        nb_h = (sous_df["character_gender"] == "m").sum()
        nb_f = (sous_df["character_gender"] == "f").sum()
        print(f"\n--- {nom} ({nb_h} H / {nb_f} F) ---")
        print(tab.round(2).to_string())
        resultats[nom] = tab

    return resultats


def analyser_films_par_genre(df, min_repliques=50):
    """Top 15 films les plus genrés et les plus paritaires."""
    print("\n" + "=" * 60)
    print("ANALYSE 2 : Classement des films par écart de genre")
    print("=" * 60)

    stats_films = []
    for titre, groupe in df.groupby("movie_title"):
        nb = len(groupe)
        if nb < min_repliques:
            continue
        pct_f = (groupe["character_gender"] == "f").mean() * 100
        stats_films.append({
            "film": titre,
            "année": int(groupe["movie_year"].iloc[0]) if pd.notna(groupe["movie_year"].iloc[0]) else None,
            "nb_répliques": nb,
            "Hommes (%)": round(100 - pct_f, 1),
            "Femmes (%)": round(pct_f, 1),
            "écart": round(abs(100 - 2 * pct_f), 1),
        })

    df_films = pd.DataFrame(stats_films)

    print("\n--- TOP 15 films les plus GENRÉS ---")
    print(df_films.sort_values("écart", ascending=False).head(15).to_string(index=False))

    print("\n--- TOP 15 films les plus PARITAIRES ---")
    print(df_films.sort_values("écart", ascending=True).head(15).to_string(index=False))

    return df_films


def analyser_profils_par_genre(df):
    """Profils normalisés par genre (chaque colonne somme à 100%)."""
    print("\n" + "=" * 60)
    print("ANALYSE 3 : Profils normalisés par genre")
    print("=" * 60)

    tab_int = pd.crosstab(
        df["intention_predite"], df["character_gender"], normalize="columns"
    ) * 100
    tab_int = tab_int.rename(columns={"m": "Hommes (%)", "f": "Femmes (%)"})
    tab_int["Écart (F−H)"] = tab_int["Femmes (%)"] - tab_int["Hommes (%)"]
    tab_int = tab_int.sort_values("Écart (F−H)", key=abs, ascending=False)
    print("\n--- Intentions ---")
    print(tab_int.round(2).to_string())

    tab_th = pd.crosstab(
        df["nom_theme"], df["character_gender"], normalize="columns"
    ) * 100
    tab_th = tab_th.rename(columns={"m": "Hommes (%)", "f": "Femmes (%)"})
    tab_th["Écart (F−H)"] = tab_th["Femmes (%)"] - tab_th["Hommes (%)"]
    tab_th = tab_th.sort_values("Écart (F−H)", key=abs, ascending=False)
    print("\n--- Thèmes ---")
    print(tab_th.round(2).to_string())

    return tab_int, tab_th


def graphique_heatmap(df):
    """Heatmap écarts F−H (%) par thème × intention."""
    themes = sorted(df["theme_dominant"].unique())
    intentions = sorted(df["intention_predite"].unique())

    matrice = pd.DataFrame(
        index=[nommer_theme(t) for t in themes], columns=intentions, dtype=float
    )

    for num_theme in themes:
        sous_df = df[df["theme_dominant"] == num_theme]
        tab = pd.crosstab(
            sous_df["intention_predite"], sous_df["character_gender"],
            normalize="columns"
        ) * 100
        for intention in intentions:
            if intention in tab.index and "f" in tab.columns and "m" in tab.columns:
                matrice.loc[nommer_theme(num_theme), intention] = (
                    tab.loc[intention, "f"] - tab.loc[intention, "m"]
                )

    matrice = matrice.fillna(0).astype(float)
    valeurs = matrice.values
    vmax = max(abs(valeurs.min()), abs(valeurs.max()))

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(valeurs, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(intentions)))
    ax.set_xticklabels(intentions, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(matrice.index)))
    ax.set_yticklabels(matrice.index, fontsize=9)

    for i in range(len(matrice.index)):
        for j in range(len(intentions)):
            val = valeurs[i, j]
            couleur = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                    fontsize=8, color=couleur)

    ax.set_title("Écart Femmes − Hommes (%) par thème et intention\n"
                 "(rouge = surreprésentation féminine, bleu = masculine)")
    plt.colorbar(im, ax=ax, label="Écart F−H (%)")
    plt.tight_layout()

    chemin = os.path.join(DOSSIER_RESULTATS, "heatmap_intention_theme_genre.png")
    plt.savefig(chemin, dpi=150)
    plt.close()
    print(f"\nGraphique sauvegardé : {chemin}")
    return matrice


def graphique_themes_par_genre(df):
    """Barplot horizontal des thèmes par genre, normalisé par colonne."""
    tab = pd.crosstab(
        df["nom_theme"], df["character_gender"], normalize="columns"
    ) * 100
    tab = tab.rename(columns={"m": "Hommes (%)", "f": "Femmes (%)"})
    tab["écart"] = tab["Femmes (%)"] - tab["Hommes (%)"]
    tab = tab.sort_values("écart")

    fig, ax = plt.subplots(figsize=(12, 7))
    tab[["Hommes (%)", "Femmes (%)"]].plot(
        kind="barh", ax=ax, color=["#4A90D9", "#E8737A"]
    )
    ax.set_xlabel("% des répliques du genre")
    ax.set_ylabel("")
    ax.set_title("Profil thématique par genre (normalisé par genre)")
    ax.legend(loc="lower right")
    plt.tight_layout()

    chemin = os.path.join(DOSSIER_RESULTATS, "themes_par_genre_nommes.png")
    plt.savefig(chemin, dpi=150)
    plt.close()
    print(f"Graphique sauvegardé : {chemin}")


def graphique_films_genre(df_films):
    """Barplot des 15 films les plus genrés et 15 plus paritaires."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for idx, (tri, titre) in enumerate([
        (False, "15 films les plus GENRÉS"),
        (True, "15 films les plus PARITAIRES"),
    ]):
        data = df_films.sort_values("écart", ascending=tri).head(15)
        ax = axes[idx]
        y = range(len(data))
        ax.barh(y, data["Hommes (%)"], color="#4A90D9", label="Hommes")
        ax.barh(y, -data["Femmes (%)"], color="#E8737A", label="Femmes")
        ax.set_yticks(y)
        ax.set_yticklabels([
            f"{r['film']} ({int(r['année'])})" if pd.notna(r['année']) else r['film']
            for _, r in data.iterrows()
        ], fontsize=8)
        ax.set_xlabel("% des répliques")
        ax.set_title(titre)
        ax.legend(loc="lower right")
        ax.axvline(x=0, color="black", linewidth=0.5)

    plt.tight_layout()
    chemin = os.path.join(DOSSIER_RESULTATS, "films_par_genre.png")
    plt.savefig(chemin, dpi=150)
    plt.close()
    print(f"Graphique sauvegardé : {chemin}")


if __name__ == "__main__":
    print("=" * 60)
    print("ANALYSES AVANCÉES — depuis resultats_complets.csv")
    print("=" * 60)

    df = charger_resultats()

    tab_int, tab_th = analyser_profils_par_genre(df)
    resultats_croises = analyser_intention_theme_genre(df)
    df_films = analyser_films_par_genre(df)

    graphique_heatmap(df)
    graphique_themes_par_genre(df)
    graphique_films_genre(df_films)

    print("\n" + "=" * 60)
    print("ANALYSES TERMINÉES")
    print("=" * 60)
