# Entraînement du Modèle A — Classification des Dialogue Acts
# Pipeline : SWDA → clean_text → TF-IDF + feature "?" → LinearSVC

import os
import sys

import joblib
from datasets import load_dataset
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.preprocessing.clean_text import preparer_dataset_swda

# ÉTAPE 1 : Chargement et prétraitement
print("=" * 60)
print("ÉTAPE 1 : Chargement et prétraitement du dataset SWDA")
print("=" * 60)

dataset = load_dataset("swda", trust_remote_code=True)
df = preparer_dataset_swda(dataset)
df["contient_point_interrogation"] = df["text"].apply(
    lambda x: 1 if "?" in str(x) else 0
)

# ÉTAPE 2 : Split train/test
print("\n" + "=" * 60)
print("ÉTAPE 2 : Séparation entraînement / test (80% / 20%)")
print("=" * 60)

X = df[["texte_nettoye", "contient_point_interrogation"]]
y = df["macro_classe"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Taille entraînement : {len(X_train)} répliques")
print(f"Taille test         : {len(X_test)} répliques")

# ÉTAPE 3 : Construction du pipeline
print("\n" + "=" * 60)
print("ÉTAPE 3 : Construction du pipeline TF-IDF + LinearSVC")
print("=" * 60)

preprocessor = ColumnTransformer(
    transformers=[
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True),
         "texte_nettoye"),
        ("features_supp", "passthrough", ["contient_point_interrogation"]),
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", LinearSVC(class_weight="balanced", max_iter=5000)),
])

print("Pipeline créé : TfidfVectorizer(ngrams 1-2, 50k features) → LinearSVC(balanced)")

# ÉTAPE 4 : Entraînement
print("\n" + "=" * 60)
print("ÉTAPE 4 : Entraînement du modèle...")
print("=" * 60)

pipeline.fit(X_train, y_train)
print("Entraînement terminé !")

# ÉTAPE 5 : Évaluation
print("\n" + "=" * 60)
print("ÉTAPE 5 : Évaluation sur le jeu de test")
print("=" * 60)

y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# ÉTAPE 6 : Sauvegarde
print("\n" + "=" * 60)
print("ÉTAPE 6 : Sauvegarde du modèle")
print("=" * 60)

chemin_modele = os.path.join(os.path.dirname(__file__), "modele_dialogue_acts.joblib")
joblib.dump(pipeline, chemin_modele)
print(f"Modèle sauvegardé dans : {chemin_modele}")
