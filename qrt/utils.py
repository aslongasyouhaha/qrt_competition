from __future__ import annotations

import os
import random
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_safe_cuda_attention() -> None:
    """Disable unstable fast-path attention kernels on some GPU/driver combos."""
    mha_backends = getattr(torch.backends, "mha", None)
    if mha_backends is not None:
        set_fastpath_enabled = getattr(mha_backends, "set_fastpath_enabled", None)
        if set_fastpath_enabled is not None:
            set_fastpath_enabled(False)

    if not torch.cuda.is_available():
        return

    cuda_backends = getattr(torch.backends, "cuda", None)
    if cuda_backends is None:
        return

    for fn_name, enabled in (
        ("enable_flash_sdp", False),
        ("enable_mem_efficient_sdp", False),
        ("enable_math_sdp", True),
    ):
        fn = getattr(cuda_backends, fn_name, None)
        if fn is not None:
            fn(enabled)


def _validate_required_columns(df: pd.DataFrame, cols: Sequence[str], df_name: str) -> None:
    missing_cols = [col for col in cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {df_name}: {missing_cols}")


def fit_standard_scaler(x_train: pd.DataFrame, cols: Sequence[str]) -> StandardScaler:
    """Fit a StandardScaler on the specified columns of x_train."""
    _validate_required_columns(x_train, cols, "x_train")
    scaler = StandardScaler()
    scaler.fit(x_train.loc[:, list(cols)])
    return scaler


def apply_standard_scaler(
    df: pd.DataFrame,
    scaler: StandardScaler,
    cols: Sequence[str],
    copy: bool = True,
) -> pd.DataFrame:
    """Apply a fitted StandardScaler to the specified columns of a DataFrame."""
    _validate_required_columns(df, cols, "df")
    result = df.copy() if copy else df
    selected_cols = list(cols)
    selected = result.loc[:, selected_cols]
    transformed = scaler.transform(selected)

    # StandardScaler emits float64 by default. Cast back to the existing block
    # dtype so pandas does not warn about incompatible assignment into float32
    # feature frames.
    target_dtype = selected.to_numpy().dtype
    transformed_frame = pd.DataFrame(
        transformed.astype(target_dtype, copy=False),
        index=result.index,
        columns=selected_cols,
    )
    result.loc[:, selected_cols] = transformed_frame
    return result


def build_gru_features(df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """Build GRU-friendly lag features from raw returns and signed volumes.

    For each lagged ``RET_i`` and ``SIGNED_VOLUME_i`` column, the function adds:
    - cross-sectional mean/std within each ``TS``
    - group mean/std within each ``(TS, GROUP)``
    - z-score relative to the ``TS`` mean/std
    - z-score relative to the ``(TS, GROUP)`` mean/std
    """
    ret_cols = [f"RET_{i}" for i in range(1, 21)]
    volume_cols = [f"SIGNED_VOLUME_{i}" for i in range(1, 21)]
    required_cols = ["TS", "GROUP", *ret_cols, *volume_cols]
    _validate_required_columns(df, required_cols, "df")

    result = df.copy() if copy else df
    feature_data: dict[str, pd.Series] = {}
    eps = 1e-8
    group_keys = [result["TS"], result["GROUP"]]

    for col in [*ret_cols, *volume_cols]:
        base_values = result[col]
        ts_mean = base_values.groupby(result["TS"]).transform("mean")
        ts_std = base_values.groupby(result["TS"]).transform("std")
        tg_mean = base_values.groupby(group_keys).transform("mean")
        tg_std = base_values.groupby(group_keys).transform("std")

        feature_data[f"TS_MEAN_{col}"] = ts_mean
        feature_data[f"TS_STD_{col}"] = ts_std
        feature_data[f"TS_GROUP_MEAN_{col}"] = tg_mean
        feature_data[f"TS_GROUP_STD_{col}"] = tg_std
        feature_data[f"Z_TS_{col}"] = (base_values - ts_mean) / (ts_std + eps)
        feature_data[f"Z_TS_GROUP_{col}"] = (base_values - tg_mean) / (tg_std + eps)

    feature_frame = pd.DataFrame(feature_data, index=result.index)
    return pd.concat([result, feature_frame], axis=1)


def build_benchmark_features(df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """Build benchmark features and group-relative variants from return history."""
    required_cols = (
        ["TS", "GROUP"]
        + [f"RET_{i}" for i in range(1, 21)]
        + [f"SIGNED_VOLUME_{i}" for i in range(1, 21)]
    )
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in df: {missing_cols}")

    result = df.copy() if copy else df
    ret_cols = [f"RET_{i}" for i in range(1, 21)]
    volume_cols = [f"SIGNED_VOLUME_{i}" for i in range(1, 21)]

    avg_windows = [2, 3, 5, 10, 15, 20]
    std_windows = [10, 20]
    range_windows = [3, 5, 10, 15, 20]
    pos_rate_windows = [3, 5, 10, 15, 20]
    volume_std_windows = [3, 5, 10, 20]
    volume_ratio_windows = [3, 5, 10, 20]
    weighted_return_windows = [3, 5, 10, 20]
    eps = 1e-8

    feature_data: dict[str, pd.Series] = {}

    def register_group_features(base_col: str, base_values: pd.Series) -> None:
        feature_data[base_col] = base_values

        ts_mean_col = f"ALLOCATIONS_{base_col}"
        ts_mean = base_values.groupby(result["TS"]).transform("mean")
        feature_data[ts_mean_col] = ts_mean

        group_keys = [result["TS"], result["GROUP"]]
        tg_mean_col = f"TS_GROUP_MEAN_{base_col}"
        tg_std_col = f"TS_GROUP_STD_{base_col}"
        tg_mean = base_values.groupby(group_keys).transform("mean")
        tg_std = base_values.groupby(group_keys).transform("std")
        feature_data[tg_mean_col] = tg_mean
        feature_data[tg_std_col] = tg_std

        rel_ts_col = f"REL_TS_{base_col}"
        rel_tg_col = f"REL_TS_GROUP_{base_col}"
        rel_tg = base_values - tg_mean
        feature_data[rel_ts_col] = base_values - ts_mean
        feature_data[rel_tg_col] = rel_tg
        feature_data[f"Z_TS_GROUP_{base_col}"] = rel_tg / (tg_std + eps)

    for window in avg_windows:
        register_group_features(
            f"AVERAGE_PERF_{window}",
            result[ret_cols[:window]].mean(axis=1),
        )

    for window in std_windows:
        register_group_features(
            f"STD_PERF_{window}",
            result[ret_cols[:window]].std(axis=1),
        )

    for window in range_windows:
        register_group_features(
            f"RANGE_PERF_{window}",
            result[ret_cols[:window]].max(axis=1) - result[ret_cols[:window]].min(axis=1),
        )

    for window in pos_rate_windows:
        register_group_features(
            f"POS_RATE_{window}",
            (result[ret_cols[:window]] > 0).mean(axis=1),
        )

    for window in [10, 20]:
        register_group_features(
            f"SHARPE_LIKE_{window}",
            feature_data[f"AVERAGE_PERF_{window}"] / (feature_data[f"STD_PERF_{window}"] + eps),
        )

    for window in avg_windows:
        register_group_features(
            f"AVERAGE_SIGNED_VOLUME_{window}",
            result[volume_cols[:window]].mean(axis=1),
        )

    for window in volume_std_windows:
        register_group_features(
            f"STD_SIGNED_VOLUME_{window}",
            result[volume_cols[:window]].std(axis=1),
        )

    for window in volume_ratio_windows:
        register_group_features(
            f"POS_SIGNED_VOLUME_RATE_{window}",
            (result[volume_cols[:window]] > 0).mean(axis=1),
        )

    for lag in range(1, 21):
        ret_col = f"RET_{lag}"
        volume_col = f"SIGNED_VOLUME_{lag}"
        register_group_features(
            f"RET_X_SIGNED_VOLUME_{lag}",
            result[ret_col] * result[volume_col],
        )
        register_group_features(
            f"RET_X_ABS_SIGNED_VOLUME_{lag}",
            result[ret_col] * result[volume_col].abs(),
        )
        register_group_features(
            f"SIGNED_VOLUME_OVER_TURNOVER_{lag}",
            result[volume_col] / (result["MEDIAN_DAILY_TURNOVER"].abs() + eps),
        )
        register_group_features(
            f"ABS_SIGNED_VOLUME_OVER_TURNOVER_{lag}",
            result[volume_col].abs() / (result["MEDIAN_DAILY_TURNOVER"].abs() + eps),
        )
        register_group_features(
            f"RET_X_SIGNED_VOLUME_OVER_TURNOVER_{lag}",
            result[ret_col] * result[volume_col] / (result["MEDIAN_DAILY_TURNOVER"].abs() + eps),
        )
        register_group_features(
            f"RET_X_ABS_SIGNED_VOLUME_OVER_TURNOVER_{lag}",
            result[ret_col] * result[volume_col].abs() / (result["MEDIAN_DAILY_TURNOVER"].abs() + eps),
        )

    for window in weighted_return_windows:
        abs_volume_sum = result[volume_cols[:window]].abs().sum(axis=1)
        signed_weighted_return = (
            (result[ret_cols[:window]] * result[volume_cols[:window]].to_numpy()).sum(axis=1)
            / (abs_volume_sum + eps)
        )
        abs_weighted_return = (
            (result[ret_cols[:window]] * result[volume_cols[:window]].abs().to_numpy()).sum(axis=1)
            / (abs_volume_sum + eps)
        )
        register_group_features(
            f"WEIGHTED_RET_SIGNED_VOLUME_{window}",
            signed_weighted_return,
        )
        register_group_features(
            f"WEIGHTED_RET_ABS_SIGNED_VOLUME_{window}",
            abs_weighted_return,
        )
        register_group_features(
            f"WEIGHTED_RET_SIGNED_VOLUME_OVER_TURNOVER_{window}",
            signed_weighted_return / (result["MEDIAN_DAILY_TURNOVER"].abs() + eps),
        )
        register_group_features(
            f"WEIGHTED_RET_ABS_SIGNED_VOLUME_OVER_TURNOVER_{window}",
            abs_weighted_return / (result["MEDIAN_DAILY_TURNOVER"].abs() + eps),
        )

    volume_spread_specs = {
        "SPREAD_AVG_SIGNED_VOLUME_3_10": ("AVERAGE_SIGNED_VOLUME_3", "AVERAGE_SIGNED_VOLUME_10"),
        "SPREAD_AVG_SIGNED_VOLUME_5_20": ("AVERAGE_SIGNED_VOLUME_5", "AVERAGE_SIGNED_VOLUME_20"),
        "SPREAD_AVG_SIGNED_VOLUME_10_20": ("AVERAGE_SIGNED_VOLUME_10", "AVERAGE_SIGNED_VOLUME_20"),
        "SPREAD_STD_SIGNED_VOLUME_5_20": ("STD_SIGNED_VOLUME_5", "STD_SIGNED_VOLUME_20"),
        "SPREAD_STD_SIGNED_VOLUME_10_20": ("STD_SIGNED_VOLUME_10", "STD_SIGNED_VOLUME_20"),
        "SPREAD_POS_SIGNED_VOLUME_RATE_3_10": ("POS_SIGNED_VOLUME_RATE_3", "POS_SIGNED_VOLUME_RATE_10"),
        "SPREAD_POS_SIGNED_VOLUME_RATE_5_20": ("POS_SIGNED_VOLUME_RATE_5", "POS_SIGNED_VOLUME_RATE_20"),
    }

    for spread_col, (left_col, right_col) in volume_spread_specs.items():
        register_group_features(spread_col, feature_data[left_col] - feature_data[right_col])

    volume_ratio_specs = {
        "RATIO_AVG_SIGNED_VOLUME_3_10": ("AVERAGE_SIGNED_VOLUME_3", "AVERAGE_SIGNED_VOLUME_10"),
        "RATIO_AVG_SIGNED_VOLUME_5_20": ("AVERAGE_SIGNED_VOLUME_5", "AVERAGE_SIGNED_VOLUME_20"),
        "RATIO_AVG_SIGNED_VOLUME_10_20": ("AVERAGE_SIGNED_VOLUME_10", "AVERAGE_SIGNED_VOLUME_20"),
        "RATIO_STD_SIGNED_VOLUME_5_20": ("STD_SIGNED_VOLUME_5", "STD_SIGNED_VOLUME_20"),
        "RATIO_STD_SIGNED_VOLUME_10_20": ("STD_SIGNED_VOLUME_10", "STD_SIGNED_VOLUME_20"),
        "RATIO_POS_SIGNED_VOLUME_RATE_3_10": ("POS_SIGNED_VOLUME_RATE_3", "POS_SIGNED_VOLUME_RATE_10"),
        "RATIO_POS_SIGNED_VOLUME_RATE_5_20": ("POS_SIGNED_VOLUME_RATE_5", "POS_SIGNED_VOLUME_RATE_20"),
    }

    for ratio_col, (left_col, right_col) in volume_ratio_specs.items():
        register_group_features(ratio_col, feature_data[left_col] / (feature_data[right_col].abs() + eps))

    spread_specs = {
        "SPREAD_AVG_3_10": ("AVERAGE_PERF_3", "AVERAGE_PERF_10"),
        "SPREAD_AVG_5_20": ("AVERAGE_PERF_5", "AVERAGE_PERF_20"),
        "SPREAD_AVG_10_20": ("AVERAGE_PERF_10", "AVERAGE_PERF_20"),
        "SPREAD_STD_10_20": ("STD_PERF_10", "STD_PERF_20"),
        "SPREAD_RANGE_3_10": ("RANGE_PERF_3", "RANGE_PERF_10"),
        "SPREAD_RANGE_5_20": ("RANGE_PERF_5", "RANGE_PERF_20"),
        "SPREAD_POS_RATE_3_10": ("POS_RATE_3", "POS_RATE_10"),
        "SPREAD_POS_RATE_5_20": ("POS_RATE_5", "POS_RATE_20"),
    }

    for spread_col, (left_col, right_col) in spread_specs.items():
        register_group_features(spread_col, feature_data[left_col] - feature_data[right_col])

    feature_frame = pd.DataFrame(feature_data, index=result.index)
    return pd.concat([result, feature_frame], axis=1)


def get_benchmark_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric model-ready benchmark columns, excluding grouping identifiers."""
    excluded_cols = {"TS", "GROUP", "ALLOCATION"}
    numeric_cols = df.select_dtypes(include="number").columns
    return [col for col in numeric_cols if col not in excluded_cols]


def _filter_columns_by_prefixes(
    cols: Sequence[str],
    include_prefixes: Sequence[str],
    exclude_prefixes: Sequence[str] | None = None,
) -> list[str]:
    excluded = tuple(exclude_prefixes or ())
    selected: list[str] = []
    for col in cols:
        if not any(col.startswith(prefix) for prefix in include_prefixes):
            continue
        if any(col.startswith(prefix) for prefix in excluded):
            continue
        selected.append(col)
    return selected


def get_catboost_global_feature_columns(
    df: pd.DataFrame,
    include_group: bool = True,
) -> tuple[list[str], list[str]]:
    """Return feature columns for the global CatBoost branch.

    The global branch keeps the full benchmark numeric feature set and can
    optionally append ``GROUP`` as a categorical feature.
    """
    numeric_feature_cols = get_benchmark_feature_columns(df)
    cat_feature_cols: list[str] = []
    if include_group:
        _validate_required_columns(df, ["GROUP"], "df")
        cat_feature_cols.append("GROUP")
    return numeric_feature_cols, cat_feature_cols


def get_catboost_group_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return a group-specialized CatBoost feature subset.

    This branch keeps raw local state plus group-aware engineered features while
    dropping purely global cross-sectional helpers such as ``ALLOCATIONS_*`` and
    ``REL_TS_*``. The goal is to make the group branch focus on within-group
    structure rather than duplicating the global branch.
    """
    numeric_feature_cols = get_benchmark_feature_columns(df)
    raw_prefixes = (
        "RET_",
        "SIGNED_VOLUME_",
        "MEDIAN_DAILY_TURNOVER",
    )
    local_summary_prefixes = (
        "AVERAGE_PERF_",
        "STD_PERF_",
        "RANGE_PERF_",
        "POS_RATE_",
        "SHARPE_LIKE_",
        "SPREAD_",
    )
    group_aware_prefixes = (
        "TS_GROUP_MEAN_",
        "TS_GROUP_STD_",
        "REL_TS_GROUP_",
        "Z_TS_GROUP_",
    )
    selected = _filter_columns_by_prefixes(
        numeric_feature_cols,
        include_prefixes=(*raw_prefixes, *local_summary_prefixes, *group_aware_prefixes),
        exclude_prefixes=("ALLOCATIONS_",),
    )
    selected = [
        col for col in selected if not (col.startswith("REL_TS_") and not col.startswith("REL_TS_GROUP_"))
    ]
    return selected


def get_gru_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return GRU feature columns generated from raw return and volume lags."""
    candidate_cols: list[str] = []
    prefixes = (
        "RET_",
        "SIGNED_VOLUME_",
        "TS_MEAN_RET_",
        "TS_STD_RET_",
        "TS_GROUP_MEAN_RET_",
        "TS_GROUP_STD_RET_",
        "Z_TS_RET_",
        "Z_TS_GROUP_RET_",
        "TS_MEAN_SIGNED_VOLUME_",
        "TS_STD_SIGNED_VOLUME_",
        "TS_GROUP_MEAN_SIGNED_VOLUME_",
        "TS_GROUP_STD_SIGNED_VOLUME_",
        "Z_TS_SIGNED_VOLUME_",
        "Z_TS_GROUP_SIGNED_VOLUME_",
    )
    for col in df.columns:
        if any(col.startswith(prefix) for prefix in prefixes):
            candidate_cols.append(col)
    return candidate_cols


def get_gru_sequence_feature_columns(df: pd.DataFrame) -> list[list[str]]:
    """Return GRU feature columns grouped by lag for sequence assembly.

    The output is ordered by lag from oldest to most recent: 20, 19, ..., 1.
    Each inner list contains the feature columns for one lag.
    """
    lag_feature_templates = [
        "RET_{lag}",
        "SIGNED_VOLUME_{lag}",
        "TS_MEAN_RET_{lag}",
        "TS_STD_RET_{lag}",
        "TS_GROUP_MEAN_RET_{lag}",
        "TS_GROUP_STD_RET_{lag}",
        "Z_TS_RET_{lag}",
        "Z_TS_GROUP_RET_{lag}",
        "TS_MEAN_SIGNED_VOLUME_{lag}",
        "TS_STD_SIGNED_VOLUME_{lag}",
        "TS_GROUP_MEAN_SIGNED_VOLUME_{lag}",
        "TS_GROUP_STD_SIGNED_VOLUME_{lag}",
        "Z_TS_SIGNED_VOLUME_{lag}",
        "Z_TS_GROUP_SIGNED_VOLUME_{lag}",
    ]

    grouped_cols: list[list[str]] = []
    for lag in range(20, 0, -1):
        lag_cols = [template.format(lag=lag) for template in lag_feature_templates]
        _validate_required_columns(df, lag_cols, "df")
        grouped_cols.append(lag_cols)
    return grouped_cols


def build_group_stratified_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    group_col: str = "GROUP",
    allocation_col: str = "ALLOCATION",
    split_group_col: str | None = None,
    fold_col: str = "fold",
) -> pd.DataFrame:
    """Build per-group folds with allocation-balanced stratification.

    Each value in ``group_col`` is split independently. By default the function
    stratifies on ``allocation_col`` and treats each row as its own group so
    allocations are distributed as evenly as possible across folds. If
    ``split_group_col`` is provided, rows sharing that value stay in the same fold.
    """
    required_cols = [group_col, allocation_col]
    if split_group_col is not None:
        required_cols.append(split_group_col)
    _validate_required_columns(df, required_cols, "df")

    result = df.copy()
    result[fold_col] = -1

    for group_value, group_df in result.groupby(group_col, sort=False):
        if len(group_df) < n_splits:
            raise ValueError(
                f"Group {group_value!r} has only {len(group_df)} rows, fewer than n_splits={n_splits}."
            )

        stratify_labels = group_df[allocation_col].astype(str)
        allocation_counts = stratify_labels.value_counts()
        if allocation_counts.min() < n_splits:
            raise ValueError(
                f"Group {group_value!r} has allocation counts below n_splits={n_splits}: "
                f"{allocation_counts[allocation_counts < n_splits].to_dict()}"
            )

        if split_group_col is None:
            split_groups = group_df.index.astype(str)
        else:
            split_groups = group_df[split_group_col].astype(str)

        sgkf = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )

        X_dummy = np.zeros(len(group_df), dtype=np.int8)
        for fold_id, (_, valid_idx) in enumerate(
            sgkf.split(X_dummy, y=stratify_labels, groups=split_groups)
        ):
            valid_index = group_df.index[valid_idx]
            result.loc[valid_index, fold_col] = fold_id

    if (result[fold_col] < 0).any():
        raise RuntimeError(f"Some rows were not assigned to {fold_col}.")

    result[fold_col] = result[fold_col].astype(int)
    return result


def build_time_balanced_folds(
    df: pd.DataFrame,
    n_splits: int = 8,
    seed: int = 42,
    ts_col: str = "TS",
    group_col: str = "GROUP",
    allocation_col: str = "ALLOCATION",
    fold_col: str = "fold",
) -> pd.DataFrame:
    """Assign whole ``TS`` buckets to folds while balancing ``(GROUP, ALLOCATION)`` counts.

    The assignment is done at the timestamp level, so all rows sharing the same
    ``TS`` stay in the same fold. A greedy objective minimizes fold imbalance
    over the observed ``(GROUP, ALLOCATION)`` combinations and total row counts.
    """
    required_cols = [ts_col, group_col, allocation_col]
    _validate_required_columns(df, required_cols, "df")

    result = df.copy()
    combo_keys = (
        result[group_col].astype(str).str.cat(result[allocation_col].astype(str), sep="__")
    )
    work = pd.DataFrame(
        {
            ts_col: result[ts_col].astype(str),
            "_combo": combo_keys,
            "_rows": 1.0,
        },
        index=result.index,
    )

    ts_combo_counts = (
        work.pivot_table(
            index=ts_col,
            columns="_combo",
            values="_rows",
            aggfunc="sum",
            fill_value=0.0,
        )
        .sort_index()
        .astype(np.float64)
    )
    ts_combo_counts["__row_count__"] = ts_combo_counts.sum(axis=1)

    ts_values = ts_combo_counts.index.to_numpy()
    if len(ts_values) < n_splits:
        raise ValueError(f"Only {len(ts_values)} unique {ts_col} values, fewer than n_splits={n_splits}.")

    counts_matrix = ts_combo_counts.to_numpy(dtype=np.float64)
    total_counts = counts_matrix.sum(axis=0)
    ideal_counts = total_counts / float(n_splits)

    rng = np.random.default_rng(seed)
    shuffle_order = rng.permutation(len(ts_values))
    row_sizes = counts_matrix[:, -1]
    ordered_indices = shuffle_order[np.argsort(-row_sizes[shuffle_order], kind="stable")]

    fold_totals = np.zeros((n_splits, counts_matrix.shape[1]), dtype=np.float64)
    fold_date_counts = np.zeros(n_splits, dtype=np.int32)
    fold_assignment: dict[str, int] = {}

    for row_idx in ordered_indices:
        row_counts = counts_matrix[row_idx]
        best_fold = 0
        best_score: float | None = None

        for fold_id in range(n_splits):
            candidate_totals = fold_totals.copy()
            candidate_totals[fold_id] += row_counts

            imbalance_score = np.square(candidate_totals - ideal_counts).sum()
            date_balance_penalty = float((fold_date_counts[fold_id] + 1) ** 2)
            score = imbalance_score + date_balance_penalty

            if best_score is None or score < best_score:
                best_score = score
                best_fold = fold_id

        fold_totals[best_fold] += row_counts
        fold_date_counts[best_fold] += 1
        fold_assignment[ts_values[row_idx]] = best_fold

    result[fold_col] = result[ts_col].astype(str).map(fold_assignment).astype(int)
    return result
