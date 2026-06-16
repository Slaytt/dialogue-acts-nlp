# Échantillonnage Cornell pour annotation humaine (Bloc 3 du cadrage Option C).
#
# Produit deux échantillons indépendants :
#   - main   : 1000 répliques stratifiées par décennie (1930-2000), équilibrées h/f émetteur
#   - orders : 150 répliques candidates ordres (heuristique regex) pour mesurer F1 ORDRE
#
# Filtrage dyadique appliqué en amont : conversations à 2 personnages,
# genres m/f connus pour les deux. Sert simultanément RQ1 (transfert SWDA→Cornell),
# RQ2 (compositionnel) et RQ3 (dyadique).
#
# Sortie : 3 CSV dans data/annotation/
#   - annotation_main.csv     (aveugle : id, contexte_avant, texte, contexte_apres, label, confiance, notes)
#   - annotation_orders.csv   (idem)
#   - metadata.csv            (fichier séparé à joindre APRÈS annotation : genres, film, etc.)

import ast
import os
import re
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.load_cornell import telecharger_cornell, parser_fichier_cornell, SEPARATEUR

SEED = 42
N_MAIN = 1000
N_ORDERS = 150

# Annotation : 3 annotateurs, bloc de recouvrement annoté par les 3 (pour le
# Krippendorff's α / Fleiss' κ), reste réparti en simple annotation.
NOMS_ANNOTATEURS = ["A", "B", "C"]
N_OVERLAP = 200
DECENNIES = [1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000]
LONGUEUR_MIN_TOKENS = 3
LONGUEUR_MAX_TOKENS = 60

# Heuristique ordres : impératif anglais courant. Volontairement large
# (l'annotateur tranchera). Capture verbes nus en tête, "don't X", "!" final,
# et quelques tournures interpersonnelles fréquentes ("tell me", "shut up").
VERBES_IMPERATIFS = (
    r"get|stop|tell|go|come|look|listen|shut|leave|wait|move|stay|sit|stand|"
    r"hold|let|give|take|put|drop|run|hurry|calm|relax|forget|remember|try|"
    r"open|close|turn|watch|hear|kiss|hit|kill|find|bring|show|help|answer|"
    r"speak|talk|say|read|write|throw|catch|push|pull|fight|fire|shoot|cut|"
    r"eat|drink|sleep|wake|stand|sign|pay|check|keep|make|do|be"
)
REGEX_ORDRE = re.compile(
    rf"^\s*(?:please\s+)?(?:({VERBES_IMPERATIFS})\b|don'?t\b|do not\b)",
    re.IGNORECASE,
)


def parser_conversations(chemin):
    """Parse movie_conversations.txt → DataFrame [conv_id, persos, line_ids]."""
    convs = []
    with open(chemin, 'r', encoding='iso-8859-1') as f:
        for i, ligne in enumerate(f):
            champs = ligne.strip().split(SEPARATEUR)
            if len(champs) != 4:
                continue
            cid1, cid2, mid, liste_str = champs
            try:
                lines = ast.literal_eval(liste_str)
            except (ValueError, SyntaxError):
                continue
            convs.append({
                "conv_id": f"conv_{i}",
                "char_a": cid1.strip(),
                "char_b": cid2.strip(),
                "movie_id": mid.strip(),
                "line_ids": lines,
            })
    df = pd.DataFrame(convs)
    print(f"  → {len(df)} conversations parsées")
    return df


def construire_table_repliques(dossier_corpus):
    """Joint lines + characters + movies + conversations → DataFrame enrichi.

    Une ligne par réplique avec :
      line_id, text, character_id, genre_emetteur, character_id_recepteur,
      genre_recepteur, movie_id, movie_title, movie_year, decennie,
      conv_id, position_dans_conv, contexte_avant, contexte_apres,
      n_persos_conv (=2 si dyadique pur)
    """
    print("\n=== Parsing fichiers Cornell ===")
    df_lines = parser_fichier_cornell(
        os.path.join(dossier_corpus, "movie_lines.txt"),
        ["line_id", "character_id", "movie_id", "character_name", "text"],
    )
    df_chars = parser_fichier_cornell(
        os.path.join(dossier_corpus, "movie_characters_metadata.txt"),
        ["character_id", "character_name", "movie_id", "movie_title", "gender", "credit_position"],
    )
    df_movies = parser_fichier_cornell(
        os.path.join(dossier_corpus, "movie_titles_metadata.txt"),
        ["movie_id", "movie_title", "movie_year", "imdb_rating", "num_imdb_votes", "genres"],
    )
    df_convs = parser_conversations(os.path.join(dossier_corpus, "movie_conversations.txt"))

    for d in (df_lines, df_chars, df_movies):
        for col in d.columns:
            d[col] = d[col].str.strip()

    df_chars["gender"] = df_chars["gender"].str.lower().str.strip()
    df_chars.loc[~df_chars["gender"].isin(["m", "f"]), "gender"] = "?"

    char_genre = dict(zip(df_chars["character_id"], df_chars["gender"]))
    movies_info = df_movies.set_index("movie_id")[["movie_title", "movie_year"]].to_dict("index")
    line_to_char = dict(zip(df_lines["line_id"], df_lines["character_id"]))
    line_to_text = dict(zip(df_lines["line_id"], df_lines["text"]))

    print("\n=== Construction table dyadique ===")
    rows = []
    for _, conv in df_convs.iterrows():
        char_a, char_b = conv["char_a"], conv["char_b"]
        g_a, g_b = char_genre.get(char_a, "?"), char_genre.get(char_b, "?")
        if g_a not in ("m", "f") or g_b not in ("m", "f"):
            continue

        lines = conv["line_ids"]
        for pos, lid in enumerate(lines):
            speaker = line_to_char.get(lid)
            if speaker is None:
                continue
            if speaker == char_a:
                g_emet, g_recep, recep = g_a, g_b, char_b
            elif speaker == char_b:
                g_emet, g_recep, recep = g_b, g_a, char_a
            else:
                continue

            text = line_to_text.get(lid, "")
            ctx_av = line_to_text.get(lines[pos - 1], "") if pos > 0 else ""
            ctx_ap = line_to_text.get(lines[pos + 1], "") if pos < len(lines) - 1 else ""

            mid = conv["movie_id"]
            minfo = movies_info.get(mid, {})
            try:
                year = float(minfo.get("movie_year", "")) if minfo.get("movie_year", "") else None
            except ValueError:
                year = None

            rows.append({
                "line_id": lid,
                "conv_id": conv["conv_id"],
                "position": pos,
                "text": text,
                "contexte_avant": ctx_av,
                "contexte_apres": ctx_ap,
                "perso_emetteur": speaker,
                "genre_emetteur": g_emet,
                "perso_recepteur": recep,
                "genre_recepteur": g_recep,
                "movie_id": mid,
                "movie_title": minfo.get("movie_title", ""),
                "movie_year": year,
            })

    df = pd.DataFrame(rows)
    df["decennie"] = (df["movie_year"] // 10 * 10).astype("Int64")
    print(f"  → {len(df):,} répliques dyadiques (conv binaires, 2 genres connus)")
    return df


def filtrer_qualite(df):
    """Filtre répliques vides, trop courtes ou trop longues."""
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df["n_tokens"] = df["text"].str.split().str.len()
    df = df[(df["n_tokens"] >= LONGUEUR_MIN_TOKENS) & (df["n_tokens"] <= LONGUEUR_MAX_TOKENS)]
    return df.reset_index(drop=True)


def echantillonner_principal(df, seed=SEED):
    """1000 répliques : 125/décennie × 8, équilibrées h/f émetteur (62 m + 63 f)."""
    df = df[df["decennie"].isin(DECENNIES)].copy()
    n_par_dec = N_MAIN // len(DECENNIES)
    n_f_par_dec = n_par_dec // 2 + n_par_dec % 2
    n_m_par_dec = n_par_dec - n_f_par_dec

    samples = []
    for dec in DECENNIES:
        sub = df[df["decennie"] == dec]
        sub_m = sub[sub["genre_emetteur"] == "m"]
        sub_f = sub[sub["genre_emetteur"] == "f"]
        n_m = min(n_m_par_dec, len(sub_m))
        n_f = min(n_f_par_dec, len(sub_f))
        if n_m < n_m_par_dec or n_f < n_f_par_dec:
            print(f"  ⚠ décennie {dec} : dispo m={len(sub_m)} f={len(sub_f)} → tiré m={n_m} f={n_f}")
        samples.append(sub_m.sample(n=n_m, random_state=seed))
        samples.append(sub_f.sample(n=n_f, random_state=seed + 1))

    df_out = pd.concat(samples, ignore_index=True)
    df_out = df_out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_out["annotation_id"] = [f"M{i:04d}" for i in range(len(df_out))]
    df_out["source_set"] = "main"
    return df_out


def echantillonner_ordres(df, deja_pris_line_ids, seed=SEED):
    """150 candidats ordres (regex impératif), équilibrés h/f émetteur, hors set principal."""
    df = df[~df["line_id"].isin(deja_pris_line_ids)].copy()
    mask_ordre = df["text"].str.contains(REGEX_ORDRE, na=False) | df["text"].str.contains(r"!\s*$", regex=True, na=False)
    candidats = df[mask_ordre].copy()
    print(f"  → {len(candidats):,} candidats détectés par regex")

    n_par_genre = N_ORDERS // 2
    sub_m = candidats[candidats["genre_emetteur"] == "m"]
    sub_f = candidats[candidats["genre_emetteur"] == "f"]
    n_m = min(n_par_genre, len(sub_m))
    n_f = min(N_ORDERS - n_m, len(sub_f))

    out = pd.concat([
        sub_m.sample(n=n_m, random_state=seed + 10),
        sub_f.sample(n=n_f, random_state=seed + 11),
    ], ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out["annotation_id"] = [f"O{i:04d}" for i in range(len(out))]
    out["source_set"] = "orders"
    return out


def ecrire_kit_annotation(df_main, df_orders, dossier_sortie, seed=SEED):
    """Produit le kit d'annotation AVEUGLE pour 3 annotateurs.

    Décisions méthodo (cf. note protocole 2026-06-16) :
    - Les deux sets (principal + ordres) sont FUSIONNÉS et mélangés : l'annotateur
      ne doit pas savoir qu'une réplique vient du set 'ordres' (sinon sur-étiquetage
      ORDRE → précision gonflée). On garde source_set seulement dans metadata.csv.
    - Ré-identification neutre (U####) : les préfixes M****/O**** trahiraient le
      source_set, on les masque et on garde le mapping dans metadata.
    - Aveugle total sur les variables d'analyse : genre, film, année, noms → tout
      dans metadata.csv, jamais dans les fichiers d'annotation.
    - Bloc de recouvrement (N_OVERLAP) annoté par les 3 → base du Krippendorff's α.

    Sorties dans dossier_sortie/ :
      - annotation_A.csv / _B.csv / _C.csv : aveugles (id_aveugle, contexte_avant,
        texte, contexte_apres, label, confiance_1_3, notes)
      - metadata.csv : à joindre APRÈS annotation via id_aveugle (orig_id, source_set,
        bloc, annotateur, genres, film, année, décennie...)
    """
    os.makedirs(dossier_sortie, exist_ok=True)

    cols_pool = [
        "annotation_id", "source_set", "line_id", "conv_id", "position",
        "perso_emetteur", "genre_emetteur", "perso_recepteur", "genre_recepteur",
        "movie_id", "movie_title", "movie_year", "decennie", "n_tokens",
        "contexte_avant", "text", "contexte_apres",
    ]
    pool = pd.concat([df_main[cols_pool], df_orders[cols_pool]], ignore_index=True)

    # Mélange + identifiant neutre (masque le source_set encodé dans M****/O****).
    pool = pool.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    pool["id_aveugle"] = [f"U{i:04d}" for i in range(len(pool))]

    # Affectation : N_OVERLAP premières répliques = recouvrement (les 3 annotateurs),
    # le reste réparti équitablement entre annotateurs (simple annotation).
    overlap = pool.iloc[:N_OVERLAP].copy()
    reste = pool.iloc[N_OVERLAP:].copy()
    parts = np.array_split(reste, len(NOMS_ANNOTATEURS))

    pool["bloc"] = "unique"
    pool.loc[pool["id_aveugle"].isin(overlap["id_aveugle"]), "bloc"] = "overlap"
    pool["annotateur"] = ""

    cols_aveugle = ["id_aveugle", "contexte_avant", "texte", "contexte_apres"]
    cols_vides = ["label", "confiance_1_3", "notes"]

    for nom, part in zip(NOMS_ANNOTATEURS, parts):
        pool.loc[pool["id_aveugle"].isin(part["id_aveugle"]), "annotateur"] = nom
        # Fichier annotateur = sa part unique + le recouvrement, re-mélangés
        # (il ne doit pas repérer quelles répliques sont le recouvrement).
        bloc = pd.concat([part, overlap], ignore_index=True)
        bloc = bloc.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        out = bloc.rename(columns={"text": "texte"})[
            ["id_aveugle", "contexte_avant", "texte", "contexte_apres"]
        ].copy()
        for c in cols_vides:
            out[c] = ""
        chemin = os.path.join(dossier_sortie, f"annotation_{nom}.csv")
        out.to_csv(chemin, index=False, encoding="utf-8")
        print(f"  → {chemin} ({len(out)} lignes : {len(part)} uniques + {len(overlap)} recouvrement)")

    cols_meta = [
        "id_aveugle", "orig_id", "source_set", "bloc", "annotateur",
        "line_id", "conv_id", "position", "perso_emetteur", "genre_emetteur",
        "perso_recepteur", "genre_recepteur", "movie_id", "movie_title",
        "movie_year", "decennie", "n_tokens",
    ]
    meta = pool.rename(columns={"annotation_id": "orig_id"})[cols_meta]
    chemin_meta = os.path.join(dossier_sortie, "metadata.csv")
    meta.to_csv(chemin_meta, index=False, encoding="utf-8")
    print(f"  → {chemin_meta} ({len(meta)} lignes — source_set/genre/film cachés ici)")


def afficher_resume(df_main, df_orders):
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES ÉCHANTILLONS")
    print("=" * 60)
    print("\n--- Set principal (1000) ---")
    print(pd.crosstab(df_main["decennie"], df_main["genre_emetteur"], margins=True))
    print("\nMatrice dyadique (émetteur → récepteur) :")
    print(pd.crosstab(df_main["genre_emetteur"], df_main["genre_recepteur"]))
    print("\n--- Set ordres (150) ---")
    print(pd.crosstab(df_orders["genre_emetteur"], df_orders["genre_recepteur"], margins=True))


def main():
    dossier_corpus = telecharger_cornell()
    df = construire_table_repliques(dossier_corpus)
    df = filtrer_qualite(df)
    print(f"\nAprès filtrage qualité : {len(df):,} répliques")

    print("\n=== Échantillonnage principal ===")
    df_main = echantillonner_principal(df)
    print(f"  → {len(df_main)} répliques tirées")

    print("\n=== Échantillonnage ordres ===")
    df_orders = echantillonner_ordres(df, set(df_main["line_id"]))
    print(f"  → {len(df_orders)} répliques tirées")

    dossier_sortie = os.path.join(os.path.dirname(__file__), "..", "..", "data", "annotation")
    print("\n=== Écriture du kit d'annotation (3 annotateurs, aveugle) ===")
    ecrire_kit_annotation(df_main, df_orders, dossier_sortie)

    afficher_resume(df_main, df_orders)


if __name__ == "__main__":
    main()
