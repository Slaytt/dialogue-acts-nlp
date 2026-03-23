# Chargement du Cornell Movie-Dialogs Corpus
# Télécharge, parse et structure le corpus en DataFrame Pandas.

import os
import zipfile
import urllib.request
import pandas as pd

URL_CORNELL = "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
DOSSIER_RAW = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
SEPARATEUR = " +++$+++ "


def telecharger_cornell(dossier_destination=None):
    """Télécharge et extrait le corpus Cornell. Retourne le chemin du dossier."""
    if dossier_destination is None:
        dossier_destination = DOSSIER_RAW

    dossier_corpus = os.path.join(dossier_destination, "cornell movie-dialogs corpus")

    fichier_test = os.path.join(dossier_corpus, "movie_lines.txt")
    if os.path.exists(fichier_test):
        print(f"Corpus Cornell déjà présent dans : {dossier_corpus}")
        return dossier_corpus

    os.makedirs(dossier_destination, exist_ok=True)

    print(f"Téléchargement du corpus Cornell (~10 Mo)...")
    print(f"URL : {URL_CORNELL}")
    donnees_zip, _ = urllib.request.urlretrieve(URL_CORNELL)

    print(f"Extraction dans : {dossier_destination}")
    with zipfile.ZipFile(donnees_zip, 'r') as z:
        z.extractall(dossier_destination)

    print("Téléchargement et extraction terminés.")
    return dossier_corpus


def parser_fichier_cornell(chemin_fichier, noms_colonnes):
    """Parse un fichier Cornell (séparateur +++$+++) → DataFrame."""
    lignes_parsees = []

    # Encodage latin-1 (pas UTF-8) pour les caractères spéciaux du corpus
    with open(chemin_fichier, 'r', encoding='iso-8859-1') as f:
        for ligne in f:
            champs = ligne.strip().split(SEPARATEUR)
            if len(champs) == len(noms_colonnes):
                lignes_parsees.append(champs)

    df = pd.DataFrame(lignes_parsees, columns=noms_colonnes)
    print(f"  → {chemin_fichier.split('/')[-1]} : {len(df)} lignes chargées")
    return df


def charger_cornell(dossier_corpus=None):
    """
    Charge et fusionne les fichiers Cornell → DataFrame unique.
    Colonnes : text, character_gender, movie_title, movie_year
    """
    if dossier_corpus is None:
        dossier_corpus = telecharger_cornell()

    print("\n=== Parsing des fichiers Cornell ===\n")

    df_lines = parser_fichier_cornell(
        os.path.join(dossier_corpus, "movie_lines.txt"),
        ["line_id", "character_id", "movie_id", "character_name", "text"]
    )

    df_characters = parser_fichier_cornell(
        os.path.join(dossier_corpus, "movie_characters_metadata.txt"),
        ["character_id", "character_name", "movie_id", "movie_title", "gender", "credit_position"]
    )

    df_movies = parser_fichier_cornell(
        os.path.join(dossier_corpus, "movie_titles_metadata.txt"),
        ["movie_id", "movie_title", "movie_year", "imdb_rating", "num_imdb_votes", "genres"]
    )

    for col in df_lines.columns:
        df_lines[col] = df_lines[col].str.strip()
    for col in df_characters.columns:
        df_characters[col] = df_characters[col].str.strip()
    for col in df_movies.columns:
        df_movies[col] = df_movies[col].str.strip()

    # Fusion répliques + genre du personnage + métadonnées film
    df_characters_reduit = df_characters[["character_id", "gender"]].drop_duplicates()
    df = df_lines.merge(df_characters_reduit, on="character_id", how="left")

    df_movies_reduit = df_movies[["movie_id", "movie_title", "movie_year"]].drop_duplicates()
    df = df.merge(df_movies_reduit, on="movie_id", how="left")

    df["gender"] = df["gender"].str.lower().str.strip()
    df.loc[~df["gender"].isin(["m", "f"]), "gender"] = "?"
    df["movie_year"] = pd.to_numeric(df["movie_year"], errors="coerce")

    df_final = df[["text", "gender", "movie_title", "movie_year"]].copy()
    df_final = df_final.rename(columns={"gender": "character_gender"})

    return df_final


def afficher_stats(df):
    """Affiche des statistiques descriptives du corpus Cornell."""
    print("\n" + "=" * 60)
    print("STATISTIQUES DU CORPUS CORNELL MOVIE-DIALOGS")
    print("=" * 60)

    print(f"\nNombre total de répliques : {len(df):,}")

    print("\n--- Distribution du genre des personnages ---")
    dist_genre = df["character_gender"].value_counts()
    for genre, count in dist_genre.items():
        pct = count / len(df) * 100
        label = {"m": "Masculin", "f": "Féminin", "?": "Inconnu"}.get(genre, genre)
        print(f"  {label:<12} : {count:>7,} répliques ({pct:.1f}%)")

    nb_films = df["movie_title"].nunique()
    print(f"\nNombre de films : {nb_films}")

    annees_valides = df["movie_year"].dropna()
    if len(annees_valides) > 0:
        print(f"Plage d'années : {int(annees_valides.min())} — {int(annees_valides.max())}")

    nb_genre_connu = len(df[df["character_gender"].isin(["m", "f"])])
    pct_connu = nb_genre_connu / len(df) * 100
    print(f"\nRépliques avec genre connu (m/f) : {nb_genre_connu:,} ({pct_connu:.1f}%)")

    print("\n--- Aperçu (5 premières répliques) ---")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    print("=== CHARGEMENT DU CORNELL MOVIE-DIALOGS CORPUS ===\n")
    df_cornell = charger_cornell()
    afficher_stats(df_cornell)
