# Projet TAL — Analyse des Biais de Genre au Cinéma

> Projet de Traitement Automatique du Langage 

**Auteurs :** Sasha Sutton & Ranzi Téo

---

## Objectif

Ce projet analyse les **dynamiques de pouvoir genrées** dans les dialogues de films en combinant deux approches complémentaires :

| Modèle | Méthode | Objectif |
|--------|---------|----------|
| **Modèle A** | Classification supervisée (LinearSVC) | Identifier les *actes de dialogue* (ordres, questions, accords...) |
| **Modèle B** | Topic modeling non-supervisé (LDA) | Extraire les *thèmes lexicaux* (violence, famille, argent...) |

Les résultats des deux modèles sont ensuite **croisés avec le genre des personnages** pour révéler des asymétries dans la façon dont hommes et femmes parlent au cinéma.

**Corpus :** [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) (~300 000 répliques, 9 000+ personnages)

**Contrainte pédagogique :** Apprentissage automatique classique uniquement (scikit-learn + spaCy) — pas de Deep Learning.

---

## Résultats

Le pipeline produit automatiquement les visualisations suivantes dans `resultats/` :

| Fichier | Description |
|---------|-------------|
| `resultats_complets.csv` | DataFrame complet (répliques + intentions + thèmes + genre) |
| `intentions_par_genre.png` | Distribution des actes de dialogue par genre |
| `themes_par_genre.png` | Distribution des thèmes LDA par genre |
| `themes_par_genre_nommes.png` | Thèmes avec labels interprétés |
| `heatmap_intention_theme_genre.png` | Heatmap croisée intention × thème × genre |
| `repartition_globale_genre.png` | Répartition globale hommes / femmes |
| `films_par_genre.png` | Nombre de films par genre cinématographique |

---

## Prérequis

- **Python 3.12** (testé sur 3.12.13 — ne pas utiliser Python 3.13, incompatible avec `datasets==2.18.0`)
- **Conda** recommandé pour isoler l'environnement

---

## Installation

```bash
# 1. Créer l'environnement
conda create -n tal python=3.12
conda activate tal

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Télécharger le modèle spaCy
python -m spacy download en_core_web_sm
```

### Dépendances principales

| Package | Rôle |
|---------|------|
| `scikit-learn` | Modèle A (LinearSVC) + Modèle B (LDA) |
| `spaCy` | Preprocessing (lemmatisation, POS tagging) |
| `pandas` | Manipulation des données |
| `datasets` | Chargement du SWDA depuis HuggingFace |
| `nltk` | Accès alternatif au corpus SWDA |
| `matplotlib` | Visualisations |

---

## Utilisation

> Les étapes 1 et 2 ne sont nécessaires que si les fichiers `.joblib` n'existent pas encore. Sinon, passer directement à l'étape 3.

### Étape 1 — Entraîner le Modèle A (classification des actes de dialogue)

Télécharge le dataset SWDA depuis HuggingFace, entraîne un LinearSVC avec TF-IDF et sauvegarde le pipeline.

```bash
python src/model_a/train_classifier.py
```

Produit : `src/model_a/modele_dialogue_acts.joblib`

### Étape 2 — Entraîner le Modèle B (extraction de thèmes LDA)

Charge le corpus Cornell, entraîne une LDA à 12 thèmes et sauvegarde le modèle.

```bash
python src/model_b/extract_topics.py
```

Produit : `src/model_b/modele_lda.joblib` et `src/model_b/vectoriseur_lda.joblib`

### Étape 3 — Lancer l'analyse croisée

Applique les deux modèles sur le Cornell corpus, croise avec le genre des personnages et génère les graphiques.

```bash
python src/analysis/merge_results.py
```

Produit : tous les fichiers dans `resultats/`

---

## Architecture du projet

```
projet_TAL/
├── data/
│   ├── raw/                          # Cornell Movie-Dialogs Corpus (brut)
│   └── processed/                    # Données préprocessées
├── src/
│   ├── preprocessing/
│   │   ├── load_cornell.py           # Parsing du corpus Cornell
│   │   └── clean_text.py             # Nettoyage + lemmatisation (spaCy)
│   ├── model_a/
│   │   ├── train_classifier.py       # Entraînement LinearSVC sur SWDA
│   │   ├── predict_intent.py         # Prédiction des actes de dialogue
│   │   └── modele_dialogue_acts.joblib
│   ├── model_b/
│   │   ├── extract_topics.py         # LDA : entraînement + assignation
│   │   ├── modele_lda.joblib
│   │   └── vectoriseur_lda.joblib
│   └── analysis/
│       ├── merge_results.py          # Analyse croisée + visualisations
│       └── analyser_csv.py           # Analyses supplémentaires sur CSV
├── notebooks/
│   ├── explo_teo.ipynb               # Exploration (Téo)
│   ├── explo_sasha.ipynb             # Exploration (Sasha)
│   └── analyse_finale.ipynb          # Notebook d'analyse finale
├── resultats/                        # Graphiques et CSV générés
├── requirements.txt
└── README.md
```

---

## Pipeline de données

```
Cornell Movie-Dialogs Corpus
        │
        ▼
  load_cornell.py          Parsing des fichiers bruts
        │
        ▼
  clean_text.py            Lemmatisation + nettoyage (spaCy)
        │
   ┌────┴────┐
   ▼         ▼
Modèle A   Modèle B
LinearSVC    LDA
(SWDA)    (12 thèmes)
   │         │
   └────┬────┘
        ▼
  merge_results.py         Croisement intentions × thèmes × genre
        │
        ▼
    resultats/             CSV + graphiques
```

---

## Thèmes LDA identifiés (Modèle B)

| # | Thème | Mots-clés |
|---|-------|-----------|
| 0 | Amour / Croyances | love, believe, old, great, business |
| 1 | Foyer / Excuses | sorry, home, understand, stop, bring |
| 2 | Travail / Société | work, man, people, new, help |
| 3 | Famille / Vie | father, mother, life, marry, feel |
| 4 | Quotidien / Mémoire | remember, day, year, forget, dead |
| 5 | Réflexion / Dialogue | think, say, maybe, mind, listen |
| 6 | Action / Mouvement | go, wait, run, away, minute |
| 7 | Espace domestique | house, room, live, die, sleep |
| 8 | Ordres / Émotions | get, tell, leave, god, stay |
| 9 | Argent / Morale | money, good, sir, bad, lie |
| 10 | Violence / Confrontation | kill, fuck, care, need, talk |
| 11 | Apparences / Rencontres | like, look, meet, ask, mean |
