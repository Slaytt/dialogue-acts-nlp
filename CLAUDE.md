# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Python virtual environment is at `.venv/` (Python 3.14). Always activate it before running anything:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Running scripts

```bash
# Run a Python script
python src/model_a/predict_intent.py

# Run a notebook as a script
python notebooks/explo_teo.py

# Launch Jupyter for notebooks
jupyter notebook notebooks/
```

## Architecture

This is a NLP/TAL project (Traitement Automatique du Langage) working on the **Switchboard Dialogue Act Corpus (SWDA)** — a dataset of spoken English dialogues labeled with dialogue acts (intentions).

The pipeline flows as follows:

```
Dataset (swda via HuggingFace or nltk.corpus.swda)
    └── src/preprocessing/clean_text.py   # Text cleaning
        ├── src/model_a/                  # Intent classification (damsl_act_tag labels)
        │       train_classifier.py       # Training
        │       predict_intent.py         # Inference
        └── src/model_b/                  # Topic extraction
                extract_topics.py
            └── src/analysis/merge_results.py  # Combine both models' outputs
```

Notebooks in `notebooks/` contain exploratory work per contributor (explo_teo, explo_sasha) and a final analysis (`analyse_finale.ipynb`).

## Dataset

The SWDA dataset can be loaded two ways:
- Via HuggingFace: `load_dataset("swda", trust_remote_code=True)` — key column is `damsl_act_tag`
- Via NLTK: `nltk.download('swda')` then `from nltk.corpus import swda` — key field is `act_tag`

The first download is slow (~several minutes). Subsequent runs use the local cache.

# Instructions globales de l'assistant
Chaque fois que tu génères un script complexe, une architecture de code, ou une réponse détaillée de plus de 15 lignes, tu dois utiliser l'outil MCP 'obsidian' (qui pointe vers /Users/slayt/Documents/Koda). 
Tu devras créer un nouveau fichier Markdown avec un titre clair dans ce dossier, y mettre au propre ta réponse structurée, puis me faire un résumé rapide ici dans le terminal en me confirmant le nom du fichier créé.

## Mode Prof (/prof)

Quand l'utilisateur commence sa phrase par `/prof` ou demande explicitement qu'on lui explique un concept :

1. **Ne JAMAIS donner la solution complète ou le code tout fait d'un coup.**
2. **Guider pas à pas** avec la méthode socratique (poser des questions pour faire réfléchir).
3. **Avant d'expliquer** : lire le fichier Obsidian `Apprentissage_Dev.md` (via MCP obsidian) pour s'adapter au niveau actuel et ne pas répéter ce qui est déjà acquis.
4. **À la fin de la session d'explication** : mettre à jour `Apprentissage_Dev.md` via MCP avec :
   - Les nouveaux concepts assimilés
   - Les difficultés du jour
   - La prochaine étape logique de progression
