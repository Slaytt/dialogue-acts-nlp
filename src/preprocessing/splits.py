"""
Split SWDA au niveau conversation (et non utterance).

Pourquoi : des énoncés voisins d'une même conversation partagent locuteurs,
sujet et style. Splitter au niveau utterance fait fuir du contexte du train
vers le test et gonfle les métriques. On groupe par conversation_no via
GroupShuffleSplit (sklearn).

Convention : SEED=42, identique à l'ancien split utterance, pour traçabilité.
"""

from sklearn.model_selection import GroupShuffleSplit

SEED = 42


def split_par_conversation(df, test_size=0.2, val_size=None, seed=SEED):
    """
    df doit contenir une colonne `conversation_no`.

    - val_size=None  → (df_train, df_test)               (≈ 80/20)
    - val_size=0.2   → (df_train, df_val, df_test)       (≈ 64/16/20)
                       val_size s'applique au df_train_full restant après
                       extraction du test.

    Pas de stratification : GroupShuffleSplit ne la supporte pas (une conv
    contient plusieurs classes). À vérifier a posteriori que chaque classe
    est présente dans chaque split (cf. verifier_distribution).
    """
    if "conversation_no" not in df.columns:
        raise ValueError(
            "df doit contenir 'conversation_no'. "
            "Regénère le cache via cache_dataset.py."
        )

    convs = df["conversation_no"].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_tr_full, idx_test = next(gss.split(df, groups=convs))
    df_tr_full = df.iloc[idx_tr_full].reset_index(drop=True)
    df_test = df.iloc[idx_test].reset_index(drop=True)

    if val_size is None:
        return df_tr_full, df_test

    convs_tr = df_tr_full["conversation_no"].to_numpy()
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    idx_tr, idx_val = next(gss2.split(df_tr_full, groups=convs_tr))
    return (
        df_tr_full.iloc[idx_tr].reset_index(drop=True),
        df_tr_full.iloc[idx_val].reset_index(drop=True),
        df_test,
    )


def verifier_distribution(splits, noms=None):
    """Affiche la taille, le nombre de conversations et la distribution des
    classes par split. Utile pour valider qu'aucune classe ne disparait."""
    import pandas as pd

    if noms is None:
        noms = [f"split_{i}" for i in range(len(splits))]

    print("\n=== Vérification des splits ===\n")
    for nom, d in zip(noms, splits):
        n_conv = d["conversation_no"].nunique()
        print(f"[{nom}] {len(d):>7} répliques | {n_conv:>4} conversations")

    print("\nDistribution des classes (proportions) :\n")
    tbl = pd.DataFrame(
        {nom: d["macro_classe"].value_counts(normalize=True) for nom, d in zip(noms, splits)}
    ).fillna(0).round(4)
    print(tbl)

    print("\nDistribution des classes (counts) :\n")
    tbl_n = pd.DataFrame(
        {nom: d["macro_classe"].value_counts() for nom, d in zip(noms, splits)}
    ).fillna(0).astype(int)
    print(tbl_n)

    manquantes = []
    for nom, d in zip(noms, splits):
        absentes = set(splits[0]["macro_classe"].unique()) - set(d["macro_classe"].unique())
        if absentes:
            manquantes.append((nom, absentes))
    if manquantes:
        print("\n⚠️  Classes manquantes :")
        for nom, c in manquantes:
            print(f"  [{nom}] : {c}")
    else:
        print("\n✓ Toutes les classes présentes dans tous les splits.")


if __name__ == "__main__":
    from src.preprocessing.cache_dataset import charger_dataset_clean

    df = charger_dataset_clean()

    print(f"\nDataFrame total : {len(df)} répliques, "
          f"{df['conversation_no'].nunique()} conversations")

    print("\n--- Split 80/20 (train / test) ---")
    df_tr, df_te = split_par_conversation(df)
    verifier_distribution([df_tr, df_te], ["train", "test"])

    print("\n\n--- Split 64/16/20 (train / val / test) ---")
    df_tr, df_va, df_te = split_par_conversation(df, val_size=0.2)
    verifier_distribution([df_tr, df_va, df_te], ["train", "val", "test"])
