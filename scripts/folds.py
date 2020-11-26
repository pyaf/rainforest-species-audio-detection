import pandas as pd
import numpy as np

df_path = "train.csv"  # original train.csv
total_folds = 5

for fold in range(5):
    df = pd.read_csv(df_path)

    good_dups = "good_duplicates.npy"
    bad_dups = "all_bad_duplicates.npy"

    bad_dups = np.load(bad_dups)
    good_dups = np.load(good_dups)
    all_dups = np.array(list(bad_dups) + list(dup_dups))

    df = df.drop(df.index[all_dups])  # remove duplicates

    kfold = StratifiedKFold(total_folds, shuffle=True, random_state=69)
    train_idx, val_idx = list(kfold.split(df["id_code"], df["diagnosis"]))[fold]
    tr_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    tr_df.to_csv(f"folds/train{fold}-5.csv")
    val_df.to_csv(f"folds/val{fold}-5.csv")

"""
# `good_duplicates.npy` contains indices of duplicates with same diagnosis. Only the duplicates, not the original image.
# `bad_duplicates.npy` contains indices of duplicates with different diagnosis, including the original image.

These npy files were generated from :https://www.kaggle.com/rishabhiitbhu/similar-duplicate-images-in-aptos-data?scriptVersionId=18820203
"""
