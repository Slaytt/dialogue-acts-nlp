# =============================================================================
# ANALYSE CROISÉE — INTENTIONS × THÈMES × GENRE
# Fichier : src/analysis/merge_results.py
# =============================================================================
# Ce fichier est le CŒUR du projet : il croise les résultats du Modèle A
# (intentions de dialogue) et du Modèle B (thèmes LDA) avec le genre des
# personnages pour mettre en évidence des biais de genre dans le cinéma.
#
# Pipeline :
#   1. Charger le corpus Cornell
#   2. Appliquer le Modèle A (prédiction d'intentions)
#   3. Appliquer le Modèle B (assignation de thèmes)
#   4. Croiser avec le genre du personnage
#   5. Calculer des statistiques et produire des visualisations

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend non-interactif pour sauvegarder les figures
import matplotlib.pyplot as plt

# Ajout du dossier racine au path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.preprocessing.load_cornell import charger_cornell
from src.preprocessing.clean_text import nettoyer_texte, MOTS_A_GARDER
from src.model_a.predict_intent import charger_modele, predire_sur_dataframe
from src.model_b.extract_topics import (
    charger_modele_lda, nettoyer_repliques, assigner_themes, afficher_themes
)
import numpy as np
import spacy

# Dossier de sortie pour les graphiques
DOSSIER_RESULTATS = os.path.join(os.path.dirname(__file__), "..", "..", "resultats")


# =============================================================================
# FONCTION 1 : construction du DataFrame complet (intentions + thèmes + genre)
# =============================================================================

def construire_dataframe_complet():
    """
    Construit le DataFrame final en appliquant les deux modèles sur le Cornell.

    Étapes :
      1. Chargement du corpus Cornell (répliques avec genre connu)
      2. Prédiction des intentions avec le Modèle A (LinearSVC)
      3. Nettoyage + assignation des thèmes avec le Modèle B (LDA)
      4. Fusion en un seul DataFrame

    Retourne :
      DataFrame avec les colonnes :
        - text : réplique brute
        - character_gender : 'm' ou 'f'
        - movie_title, movie_year
        - intention_predite : macro-classe du Modèle A
        - texte_nettoye : réplique nettoyée par spaCy
        - theme_dominant : numéro du thème LDA
    """
    # --- Étape 1 : chargement du corpus Cornell ---
    print("=" * 60)
    print("ÉTAPE 1 : Chargement du corpus Cornell")
    print("=" * 60)
    df = charger_cornell()

    # On garde uniquement les répliques avec genre connu
    df = df[df["character_gender"].isin(["m", "f"])].copy()
    df = df.reset_index(drop=True)
    print(f"\nRépliques avec genre connu : {len(df)}")

    # --- Étape 2 : prédiction des intentions (Modèle A) ---
    print("\n" + "=" * 60)
    print("ÉTAPE 2 : Prédiction des intentions (Modèle A)")
    print("=" * 60)
    pipeline_a = charger_modele()
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    # Réhabilitation des stop words discriminants (identique à l'entraînement)
    for mot in MOTS_A_GARDER:
        nlp.vocab[mot].is_stop = False
    nlp.vocab["n't"].is_stop = False
    df = predire_sur_dataframe(df, "text", pipeline_a, nlp)

    # Suppression des répliques classées "VIDE" (texte vide après nettoyage)
    nb_vides = (df["intention_predite"] == "VIDE").sum()
    df = df[df["intention_predite"] != "VIDE"].copy()
    print(f"Répliques VIDE supprimées : {nb_vides}")

    # --- Étape 3 : assignation des thèmes (Modèle B) ---
    print("\n" + "=" * 60)
    print("ÉTAPE 3 : Assignation des thèmes (Modèle B — LDA)")
    print("=" * 60)
    modele_lda, vectoriseur = charger_modele_lda()

    # Nettoyage des répliques pour le LDA
    textes_propres = nettoyer_repliques(df, col_texte="text")

    # On ne garde que les répliques non-vides après nettoyage
    df = df.loc[textes_propres.index].copy()
    df["texte_nettoye"] = textes_propres

    # Assignation du thème dominant
    themes = assigner_themes(modele_lda, vectoriseur, textes_propres)
    df["theme_dominant"] = themes.values

    print(f"\nDataFrame final : {len(df)} répliques")
    return df


# =============================================================================
# FONCTION 2 : analyse des intentions par genre
# =============================================================================

def analyser_intentions_par_genre(df):
    """
    Calcule la distribution des intentions par genre et affiche les résultats.

    Pour chaque intention (ORDRE, QUESTION, POLITESSE...), on calcule
    le pourcentage prononcé par des hommes vs des femmes.

    Paramètre :
      df (DataFrame) : le DataFrame complet (sortie de construire_dataframe_complet)

    Retourne :
      DataFrame : tableau croisé intentions × genre (en pourcentages)
    """
    print("\n" + "=" * 60)
    print("ANALYSE 1 : Distribution des intentions par genre")
    print("=" * 60)

    # Tableau croisé : pour chaque genre, quel % de ses répliques va dans chaque intention
    # normalize="columns" → normalisation par genre (chaque colonne somme à 100%)
    # Cela élimine le biais de surreprésentation masculine : on compare les PROFILS
    tableau = pd.crosstab(
        df["intention_predite"],
        df["character_gender"],
        normalize="columns"  # normalisation par genre → profils comparables
    ) * 100

    # Renommage des colonnes pour la lisibilité
    tableau = tableau.rename(columns={"m": "Hommes (%)", "f": "Femmes (%)"})

    # Ajout du nombre total de répliques par intention
    tableau["Total"] = df["intention_predite"].value_counts()

    # Tri par écart le plus marqué (différence hommes-femmes)
    tableau["Écart"] = abs(tableau["Hommes (%)"] - tableau["Femmes (%)"])
    tableau = tableau.sort_values("Écart", ascending=False)

    print("\n" + tableau.round(1).to_string())

    return tableau


# =============================================================================
# FONCTION 3 : analyse des thèmes par genre
# =============================================================================

def analyser_themes_par_genre(df):
    """
    Calcule la distribution des thèmes LDA par genre.

    Pour chaque thème (0 à 11), on calcule le pourcentage de répliques
    prononcées par des hommes vs des femmes.

    Paramètre :
      df (DataFrame) : le DataFrame complet

    Retourne :
      DataFrame : tableau croisé thèmes × genre (en pourcentages)
    """
    print("\n" + "=" * 60)
    print("ANALYSE 2 : Distribution des thèmes par genre")
    print("=" * 60)

    # Tableau croisé : pour chaque genre, quel % de ses répliques va dans chaque thème
    # normalize="columns" → normalisation par genre (profils comparables)
    tableau = pd.crosstab(
        df["theme_dominant"],
        df["character_gender"],
        normalize="columns"
    ) * 100

    tableau = tableau.rename(columns={"m": "Hommes (%)", "f": "Femmes (%)"})
    tableau["Total"] = df["theme_dominant"].value_counts()
    tableau["Écart"] = abs(tableau["Hommes (%)"] - tableau["Femmes (%)"])
    tableau = tableau.sort_values("Écart", ascending=False)

    print("\n" + tableau.round(1).to_string())

    return tableau


# =============================================================================
# FONCTION 4 : analyse temporelle (par décennie)
# =============================================================================

def analyser_evolution_temporelle(df):
    """
    Analyse l'évolution de la répartition des intentions par genre au fil du temps.

    On regroupe les films par décennie et on observe si les biais
    évoluent (ex: les femmes reçoivent-elles plus d'ORDRES dans les films récents ?).

    Paramètre :
      df (DataFrame) : le DataFrame complet

    Retourne :
      DataFrame : intentions × genre × décennie
    """
    print("\n" + "=" * 60)
    print("ANALYSE 3 : Évolution temporelle (par décennie)")
    print("=" * 60)

    # Création de la colonne décennie
    df_temp = df.dropna(subset=["movie_year"]).copy()
    df_temp["decennie"] = (df_temp["movie_year"] // 10 * 10).astype(int)

    # On ne garde que les décennies avec assez de données
    counts_dec = df_temp["decennie"].value_counts()
    decennies_valides = counts_dec[counts_dec > 1000].index
    df_temp = df_temp[df_temp["decennie"].isin(decennies_valides)]

    # Pour chaque décennie, proportion de femmes par intention
    resultats = []
    for dec in sorted(df_temp["decennie"].unique()):
        sous_df = df_temp[df_temp["decennie"] == dec]
        for intention in sorted(sous_df["intention_predite"].unique()):
            sous_intention = sous_df[sous_df["intention_predite"] == intention]
            pct_femmes = (sous_intention["character_gender"] == "f").mean() * 100
            resultats.append({
                "decennie": dec,
                "intention": intention,
                "pct_femmes": round(pct_femmes, 1),
                "nb_repliques": len(sous_intention)
            })

    df_evol = pd.DataFrame(resultats)

    # Affichage pivot : décennies en lignes, intentions en colonnes
    pivot = df_evol.pivot(index="decennie", columns="intention", values="pct_femmes")
    print("\n% de répliques prononcées par des FEMMES, par décennie :\n")
    print(pivot.round(1).to_string())

    return df_evol


# =============================================================================
# FONCTION 5 : génération des graphiques
# =============================================================================

def generer_graphiques(df, tab_intentions, tab_themes):
    """
    Génère et sauvegarde les graphiques de l'analyse croisée.

    Graphiques produits :
      1. Barplot : intentions par genre
      2. Barplot : thèmes par genre
      3. Proportion globale hommes/femmes

    Paramètres :
      df              : le DataFrame complet
      tab_intentions  : tableau croisé intentions × genre
      tab_themes      : tableau croisé thèmes × genre
    """
    # Création du dossier de résultats
    os.makedirs(DOSSIER_RESULTATS, exist_ok=True)

    # --- Graphique 1 : intentions par genre ---
    fig, ax = plt.subplots(figsize=(12, 6))
    tab_plot = tab_intentions[["Hommes (%)", "Femmes (%)"]].sort_index()
    tab_plot.plot(kind="barh", ax=ax, color=["#4A90D9", "#E8737A"])
    ax.set_xlabel("Pourcentage (%)")
    ax.set_ylabel("Intention")
    ax.set_title("Distribution des intentions par genre — Cornell Movie-Dialogs")
    ax.legend(loc="lower right")
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5)  # ligne de parité
    plt.tight_layout()
    chemin1 = os.path.join(DOSSIER_RESULTATS, "intentions_par_genre.png")
    plt.savefig(chemin1, dpi=150)
    plt.close()
    print(f"\nGraphique sauvegardé : {chemin1}")

    # --- Graphique 2 : thèmes par genre ---
    fig, ax = plt.subplots(figsize=(12, 6))
    tab_plot2 = tab_themes[["Hommes (%)", "Femmes (%)"]].sort_index()
    tab_plot2.index = [nommer_theme(int(i)) for i in tab_plot2.index]
    tab_plot2.plot(kind="barh", ax=ax, color=["#4A90D9", "#E8737A"])
    ax.set_xlabel("Pourcentage (%)")
    ax.set_ylabel("Thème LDA")
    ax.set_title("Distribution des thèmes par genre — Cornell Movie-Dialogs")
    ax.legend(loc="lower right")
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    chemin2 = os.path.join(DOSSIER_RESULTATS, "themes_par_genre.png")
    plt.savefig(chemin2, dpi=150)
    plt.close()
    print(f"Graphique sauvegardé : {chemin2}")

    # --- Graphique 3 : proportion globale ---
    fig, ax = plt.subplots(figsize=(6, 6))
    dist = df["character_gender"].value_counts()
    labels = ["Hommes", "Femmes"]
    couleurs = ["#4A90D9", "#E8737A"]
    ax.pie(dist.values, labels=labels, colors=couleurs, autopct="%1.1f%%",
           startangle=90, textprops={"fontsize": 14})
    ax.set_title("Répartition globale des répliques par genre")
    plt.tight_layout()
    chemin3 = os.path.join(DOSSIER_RESULTATS, "repartition_globale_genre.png")
    plt.savefig(chemin3, dpi=150)
    plt.close()
    print(f"Graphique sauvegardé : {chemin3}")


# =============================================================================
# NOMS DES THÈMES LDA (interprétation manuelle des top mots)
# =============================================================================
# Ces noms sont attribués en regardant les 10 mots les plus représentatifs
# de chaque thème (via afficher_themes()). C'est subjectif mais nécessaire
# pour rendre les résultats lisibles.

NOMS_THEMES = {
    0: "Amour / Croyances",       # love, believe, old, great, word, business
    1: "Foyer / Excuses",         # sorry, home, understand, stop, bring
    2: "Travail / Société",       # work, man, people, new, help, miss
    3: "Famille / Vie",           # father, mother, life, marry, feel
    4: "Quotidien / Mémoire",     # remember, day, year, forget, dead, pay
    5: "Réflexion / Dialogue",    # think, say, maybe, mind, listen, hope
    6: "Action / Mouvement",      # go, wait, run, away, minute, guy, girl
    7: "Espace domestique",       # house, room, live, die, sleep, dad
    8: "Ordres / Émotions",       # get, tell, leave, god, stay, sit
    9: "Argent / Morale",         # money, good, sir, bad, lie, hurt, damn
    10: "Violence / Confrontation", # kill, fuck, care, need, talk, try
    11: "Apparences / Rencontres",  # like, look, meet, ask, mean, sound
}


def nommer_theme(num):
    """Retourne le nom lisible d'un thème LDA à partir de son numéro."""
    return NOMS_THEMES.get(num, f"Thème {num}")


# =============================================================================
# ANALYSE 5 : croisement intention × thème × genre
# =============================================================================

def analyser_intention_theme_genre(df):
    """
    Pour chaque thème, compare le profil d'intentions des hommes vs femmes.

    Permet de répondre à : "Dans le thème Famille, qui donne les ORDRES ?"
    La normalisation par genre (chaque genre somme à 100% dans un thème)
    élimine le biais de surreprésentation masculine.

    Paramètre :
      df (DataFrame) : le DataFrame complet

    Retourne :
      dict : {nom_theme: DataFrame avec colonnes Hommes%, Femmes%, Écart}
    """
    print("\n" + "=" * 60)
    print("ANALYSE : Intentions × Thèmes × Genre")
    print("=" * 60)

    resultats = {}

    for num_theme in sorted(df["theme_dominant"].unique()):
        nom = nommer_theme(num_theme)
        sous_df = df[df["theme_dominant"] == num_theme]

        # Profil d'intentions par genre dans ce thème
        tab = pd.crosstab(
            sous_df["intention_predite"],
            sous_df["character_gender"],
            normalize="columns"
        ) * 100

        tab = tab.rename(columns={"m": "Hommes (%)", "f": "Femmes (%)"})
        tab["Écart"] = tab["Femmes (%)"] - tab["Hommes (%)"]
        tab = tab.sort_values("Écart", key=abs, ascending=False)

        nb_h = (sous_df["character_gender"] == "m").sum()
        nb_f = (sous_df["character_gender"] == "f").sum()

        print(f"\n--- Thème {num_theme} : {nom} ({nb_h} H / {nb_f} F) ---")
        print(tab.round(2).to_string())

        resultats[nom] = tab

    return resultats


# =============================================================================
# ANALYSE 6 : classement des films par écart de genre
# =============================================================================

def analyser_films_par_genre(df, min_repliques=50):
    """
    Classe les films selon leur écart de répartition hommes/femmes.

    Permet d'identifier les films les plus genrés (presque que des hommes)
    et les plus paritaires.

    Paramètres :
      df (DataFrame)       : le DataFrame complet
      min_repliques (int)  : seuil minimum de répliques pour inclure un film

    Retourne :
      DataFrame : un film par ligne, avec colonnes H%, F%, écart, nb répliques
    """
    print("\n" + "=" * 60)
    print("ANALYSE : Classement des films par écart de genre")
    print("=" * 60)

    # Calcul par film
    stats_films = []
    for titre, groupe in df.groupby("movie_title"):
        nb = len(groupe)
        if nb < min_repliques:
            continue
        pct_f = (groupe["character_gender"] == "f").mean() * 100
        pct_h = 100 - pct_f
        annee = groupe["movie_year"].iloc[0]
        stats_films.append({
            "film": titre,
            "année": int(annee) if pd.notna(annee) else None,
            "nb_répliques": nb,
            "Hommes (%)": round(pct_h, 1),
            "Femmes (%)": round(pct_f, 1),
            "écart": round(abs(pct_h - pct_f), 1),
        })

    df_films = pd.DataFrame(stats_films)

    # Top 15 films les plus genrés (écart le plus grand)
    top_genres = df_films.sort_values("écart", ascending=False).head(15)
    print("\n--- TOP 15 films les plus GENRÉS (écart H/F le plus grand) ---")
    print(top_genres.to_string(index=False))

    # Top 15 films les plus paritaires
    top_paritaires = df_films.sort_values("écart", ascending=True).head(15)
    print("\n--- TOP 15 films les plus PARITAIRES (écart H/F le plus petit) ---")
    print(top_paritaires.to_string(index=False))

    return df_films


# =============================================================================
# GRAPHIQUES SUPPLÉMENTAIRES
# =============================================================================

def generer_graphique_intention_theme_genre(df):
    """
    Génère une heatmap montrant l'écart F-H pour chaque combinaison
    intention × thème. Rouge = surreprésentation féminine, bleu = masculine.
    """
    # Construction de la matrice écart par thème × intention
    themes = sorted(df["theme_dominant"].unique())
    intentions = sorted(df["intention_predite"].unique())

    matrice_ecart = pd.DataFrame(index=[nommer_theme(t) for t in themes],
                                  columns=intentions, dtype=float)

    for num_theme in themes:
        sous_df = df[df["theme_dominant"] == num_theme]
        tab = pd.crosstab(
            sous_df["intention_predite"],
            sous_df["character_gender"],
            normalize="columns"
        ) * 100
        for intention in intentions:
            if intention in tab.index and "f" in tab.columns and "m" in tab.columns:
                ecart = tab.loc[intention, "f"] - tab.loc[intention, "m"]
                matrice_ecart.loc[nommer_theme(num_theme), intention] = ecart

    matrice_ecart = matrice_ecart.fillna(0).astype(float)

    # Heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    valeurs = matrice_ecart.values
    vmax = max(abs(valeurs.min()), abs(valeurs.max()))

    im = ax.imshow(valeurs, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(intentions)))
    ax.set_xticklabels(intentions, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(matrice_ecart.index)))
    ax.set_yticklabels(matrice_ecart.index, fontsize=9)

    # Afficher les valeurs dans les cellules
    for i in range(len(matrice_ecart.index)):
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

    return matrice_ecart


# =============================================================================
# EXÉCUTION PRINCIPALE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ANALYSE CROISÉE — INTENTIONS × THÈMES × GENRE")
    print("=" * 60)

    # --- Construction du DataFrame complet ---
    df_complet = construire_dataframe_complet()

    # --- Analyses statistiques ---
    tab_intentions = analyser_intentions_par_genre(df_complet)
    tab_themes = analyser_themes_par_genre(df_complet)
    df_evol = analyser_evolution_temporelle(df_complet)

    # --- Nouvelles analyses ---
    resultats_croises = analyser_intention_theme_genre(df_complet)
    df_films = analyser_films_par_genre(df_complet)

    # --- Génération des graphiques ---
    generer_graphiques(df_complet, tab_intentions, tab_themes)
    matrice_ecart = generer_graphique_intention_theme_genre(df_complet)

    # --- Sauvegarde du DataFrame complet pour le notebook final ---
    chemin_csv = os.path.join(DOSSIER_RESULTATS, "resultats_complets.csv")
    os.makedirs(DOSSIER_RESULTATS, exist_ok=True)
    df_complet.to_csv(chemin_csv, index=False)
    print(f"\nDataFrame complet sauvegardé : {chemin_csv}")

    print("\n" + "=" * 60)
    print("ANALYSE CROISÉE TERMINÉE")
    print("=" * 60)
