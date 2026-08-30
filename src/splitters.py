from __future__ import annotations

import pandas as pd


def group_train_test_split(
    df: pd.DataFrame, group_cols: list[str], label_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split on group columns to avoid leakage between train and test.

    Groups are defined by the unique combinations of group_cols.
    A deterministic hash split is used for reproducibility.
    """

    if not group_cols:
        raise ValueError("group_cols must be a non-empty list")
    missing_cols = [c for c in group_cols + [label_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"missing columns: {missing_cols}")

    # Deterministic 80/20 group split using a stable hash.
    test_frac = 0.2
    group_keys = df[group_cols].drop_duplicates().reset_index(drop=True)
    n_groups = len(group_keys)
    if n_groups < 2:
        # Not enough groups to split; fall back to a single train set.
        return df.copy(), df.iloc[0:0].copy()

    group_hash = pd.util.hash_pandas_object(group_keys, index=False).astype("uint64")
    cutoff = int(group_hash.max() * (1 - test_frac))
    group_keys["__is_test__"] = group_hash > cutoff

    df = df.merge(group_keys, on=group_cols, how="left")
    test_df = df[df["__is_test__"]].drop(columns=["__is_test__"])
    train_df = df[~df["__is_test__"]].drop(columns=["__is_test__"])

    # Ensure both splits are non-empty; if not, use a deterministic group split.
    if len(test_df) == 0 or len(train_df) == 0:
        group_keys = group_keys.sort_values(group_cols).reset_index(drop=True)
        n_test_groups = max(1, int(round(n_groups * test_frac)))
        test_groups = group_keys.tail(n_test_groups)[group_cols]
        test_df = df.merge(test_groups, on=group_cols, how="inner")
        train_df = df.merge(test_groups, on=group_cols, how="left", indicator=True)
        train_df = train_df[train_df["_merge"] == "left_only"].drop(columns=["_merge"])

    return train_df, test_df
