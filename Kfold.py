from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import pandas as pd
import os

mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=0)


def kfold(df):
    X = df['img_path']
    y = df.iloc[:, 2:].values

    output_directory = "fold_data"
    os.makedirs(output_directory, exist_ok=True)

    for fold_idx, (train_index, test_index) in enumerate(mskf.split(X, y), 1):
        fold_train_df = df.iloc[train_index]
        fold_test_df = df.iloc[test_index]
        
        train_filename = os.path.join(output_directory, f"fold_{fold_idx}_train.csv")
        test_filename = os.path.join(output_directory, f"fold_{fold_idx}_test.csv")

        fold_train_df.to_csv(train_filename, index=False)
        fold_test_df.to_csv(test_filename, index=False)

        print(f"Fold {fold_idx} - Train saved to: {train_filename}")
        print(f"Fold {fold_idx} - Test saved to: {test_filename}")
