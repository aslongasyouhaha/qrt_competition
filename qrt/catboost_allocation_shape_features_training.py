from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from utils import (
    apply_standard_scaler,
    build_benchmark_features,
    fit_standard_scaler,
    get_catboost_global_feature_columns,
    seed_everything,
)


SEED_VALUES = (42, 52, 62)
N_SPLITS = 5
FILLNA_VALUE = 0.0
SCALE_FEATURES = True

CATBOOST_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.01,
    "depth": 3,
    "l2_leaf_reg": 3.0,
    "loss_function": "RMSE",
    "verbose": False,
}


SHAPE_FEATURE_COLUMNS = [
    "RET_SLOPE_3",
    "RET_SLOPE_5",
    "RET_SLOPE_10",
    "RET_SLOPE_20",
    "SIGNED_VOLUME_SLOPE_3",
    "SIGNED_VOLUME_SLOPE_5",
    "SIGNED_VOLUME_SLOPE_10",
    "SIGNED_VOLUME_SLOPE_20",
    "RET_SIGN_RUN_LAST_20",
    "SIGNED_VOLUME_SIGN_RUN_LAST_20",
    "RET_SIGN_CHANGES_20",
    "SIGNED_VOLUME_SIGN_CHANGES_20",
    "RET_SKEW_10",
    "RET_SKEW_20",
    "RET_KURT_10",
    "RET_KURT_20",
    "SIGNED_VOLUME_SKEW_10",
    "SIGNED_VOLUME_SKEW_20",
    "SIGNED_VOLUME_KURT_10",
    "SIGNED_VOLUME_KURT_20",
    "RET_AUTOCORR_LAG1_10",
    "RET_AUTOCORR_LAG1_20",
    "SIGNED_VOLUME_AUTOCORR_LAG1_10",
    "SIGNED_VOLUME_AUTOCORR_LAG1_20",
]


def directional_accuracy(scores, targets) -> float:
    preds = (np.asarray(scores) >= 0.0).astype(int)
    target_binary = (np.asarray(targets) >= 0.0).astype(int)
    return float((preds == target_binary).mean())


def build_allocation_time_folds(
    allocation_df: pd.DataFrame,
    n_splits: int = 5,
    ts_col: str = "TS",
    fold_col: str = "fold",
) -> pd.Series:
    unique_ts = sorted(allocation_df[ts_col].dropna().unique().tolist())
    if len(unique_ts) < n_splits:
        raise ValueError(f"Not enough unique TS to build {n_splits} folds: only {len(unique_ts)}")
    fold_map = {ts_value: idx % n_splits for idx, ts_value in enumerate(unique_ts)}
    return allocation_df[ts_col].map(fold_map).astype(int).rename(fold_col)


def _rowwise_slope(matrix: np.ndarray) -> np.ndarray:
    x = np.arange(matrix.shape[1], dtype=np.float32)
    x_centered = x - x.mean()
    denom = float(np.square(x_centered).sum())
    y_centered = matrix - matrix.mean(axis=1, keepdims=True)
    return (y_centered * x_centered).sum(axis=1) / denom


def _rowwise_last_sign_run(matrix: np.ndarray) -> np.ndarray:
    signs = np.sign(matrix)
    result = np.zeros(matrix.shape[0], dtype=np.float32)
    for idx, row in enumerate(signs):
        last_sign = row[-1]
        if last_sign == 0:
            result[idx] = 0.0
            continue
        run_length = 1
        for value in row[-2::-1]:
            if value == last_sign:
                run_length += 1
            else:
                break
        result[idx] = float(run_length)
    return result


def _rowwise_sign_changes(matrix: np.ndarray) -> np.ndarray:
    signs = np.sign(matrix)
    nonzero = np.where(signs == 0, np.nan, signs)
    changes = np.zeros(matrix.shape[0], dtype=np.float32)
    for idx, row in enumerate(nonzero):
        valid = row[~np.isnan(row)]
        if len(valid) <= 1:
            changes[idx] = 0.0
        else:
            changes[idx] = float(np.sum(valid[1:] != valid[:-1]))
    return changes


def _rowwise_autocorr_lag1(matrix: np.ndarray) -> np.ndarray:
    left = matrix[:, :-1]
    right = matrix[:, 1:]
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(np.sum(left_centered**2, axis=1) * np.sum(right_centered**2, axis=1)) + 1e-8
    return numerator / denominator


def build_shape_features(df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    result = df.copy() if copy else df
    ret_cols = [f"RET_{i}" for i in range(1, 21)]
    vol_cols = [f"SIGNED_VOLUME_{i}" for i in range(1, 21)]
    ret_matrix = result.loc[:, ret_cols].to_numpy(dtype=np.float32)
    vol_matrix = result.loc[:, vol_cols].to_numpy(dtype=np.float32)

    feature_data: dict[str, np.ndarray | pd.Series] = {}

    for window in (3, 5, 10, 20):
        ret_window = ret_matrix[:, :window]
        vol_window = vol_matrix[:, :window]
        feature_data[f"RET_SLOPE_{window}"] = _rowwise_slope(ret_window).astype(np.float32)
        feature_data[f"SIGNED_VOLUME_SLOPE_{window}"] = _rowwise_slope(vol_window).astype(np.float32)

    feature_data["RET_SIGN_RUN_LAST_20"] = _rowwise_last_sign_run(ret_matrix).astype(np.float32)
    feature_data["SIGNED_VOLUME_SIGN_RUN_LAST_20"] = _rowwise_last_sign_run(vol_matrix).astype(np.float32)
    feature_data["RET_SIGN_CHANGES_20"] = _rowwise_sign_changes(ret_matrix).astype(np.float32)
    feature_data["SIGNED_VOLUME_SIGN_CHANGES_20"] = _rowwise_sign_changes(vol_matrix).astype(np.float32)

    ret_10 = result.loc[:, ret_cols[:10]]
    ret_20 = result.loc[:, ret_cols]
    vol_10 = result.loc[:, vol_cols[:10]]
    vol_20 = result.loc[:, vol_cols]

    feature_data["RET_SKEW_10"] = ret_10.skew(axis=1).fillna(0.0).astype(np.float32)
    feature_data["RET_SKEW_20"] = ret_20.skew(axis=1).fillna(0.0).astype(np.float32)
    feature_data["RET_KURT_10"] = ret_10.kurt(axis=1).fillna(0.0).astype(np.float32)
    feature_data["RET_KURT_20"] = ret_20.kurt(axis=1).fillna(0.0).astype(np.float32)
    feature_data["SIGNED_VOLUME_SKEW_10"] = vol_10.skew(axis=1).fillna(0.0).astype(np.float32)
    feature_data["SIGNED_VOLUME_SKEW_20"] = vol_20.skew(axis=1).fillna(0.0).astype(np.float32)
    feature_data["SIGNED_VOLUME_KURT_10"] = vol_10.kurt(axis=1).fillna(0.0).astype(np.float32)
    feature_data["SIGNED_VOLUME_KURT_20"] = vol_20.kurt(axis=1).fillna(0.0).astype(np.float32)

    feature_data["RET_AUTOCORR_LAG1_10"] = _rowwise_autocorr_lag1(ret_matrix[:, :10]).astype(np.float32)
    feature_data["RET_AUTOCORR_LAG1_20"] = _rowwise_autocorr_lag1(ret_matrix).astype(np.float32)
    feature_data["SIGNED_VOLUME_AUTOCORR_LAG1_10"] = _rowwise_autocorr_lag1(vol_matrix[:, :10]).astype(np.float32)
    feature_data["SIGNED_VOLUME_AUTOCORR_LAG1_20"] = _rowwise_autocorr_lag1(vol_matrix).astype(np.float32)

    return pd.concat([result, pd.DataFrame(feature_data, index=result.index)], axis=1)


def prepare_frames(
    x_train_path: str | Path,
    y_train_path: str | Path,
    x_test_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    x_train = pd.read_csv(x_train_path, index_col="ROW_ID")
    y_train = pd.read_csv(y_train_path, index_col="ROW_ID")
    x_test = pd.read_csv(x_test_path, index_col="ROW_ID")

    x_train = build_shape_features(build_benchmark_features(x_train))
    x_test = build_shape_features(build_benchmark_features(x_test))

    numeric_feature_cols, _cat_feature_names = get_catboost_global_feature_columns(
        x_train,
        include_group=True,
    )
    feature_cols = list(dict.fromkeys([*numeric_feature_cols, *SHAPE_FEATURE_COLUMNS]))

    scaler_frame = x_train.copy()
    scaler_frame.loc[:, feature_cols] = scaler_frame.loc[:, feature_cols].fillna(FILLNA_VALUE)
    scaler = fit_standard_scaler(scaler_frame, feature_cols)

    prepared_train = x_train.copy()
    prepared_test = x_test.copy()
    prepared_train.loc[:, feature_cols] = prepared_train.loc[:, feature_cols].fillna(FILLNA_VALUE)
    prepared_test.loc[:, feature_cols] = prepared_test.loc[:, feature_cols].fillna(FILLNA_VALUE)

    if SCALE_FEATURES:
        prepared_train = apply_standard_scaler(prepared_train, scaler, feature_cols, copy=False)
        prepared_test = apply_standard_scaler(prepared_test, scaler, feature_cols, copy=False)

    prepared_train["GROUP"] = prepared_train["GROUP"].astype(str)
    prepared_train["ALLOCATION"] = prepared_train["ALLOCATION"].astype(str)
    prepared_test["GROUP"] = prepared_test["GROUP"].astype(str)
    prepared_test["ALLOCATION"] = prepared_test["ALLOCATION"].astype(str)
    return prepared_train, y_train, prepared_test, feature_cols


def fit_catboost_regressor(train_x: pd.DataFrame, train_y: pd.Series, random_seed: int):
    model = CatBoostRegressor(**{**CATBOOST_PARAMS, "random_seed": random_seed})
    model.fit(train_x, train_y)
    return model


def train_catboost_allocation_shape_features(
    x_train_path: str | Path,
    y_train_path: str | Path,
    x_test_path: str | Path,
    output_dir: str | Path,
    preds_output_path: str | Path | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED_VALUES[0])

    prepared_train, y_train, prepared_test, feature_cols = prepare_frames(
        x_train_path=x_train_path,
        y_train_path=y_train_path,
        x_test_path=x_test_path,
    )

    oof_scores = pd.DataFrame(index=prepared_train.index)
    test_scores = pd.DataFrame(index=prepared_test.index)
    fold_metrics = []
    manifest_rows = []

    allocation_values = sorted(prepared_train["ALLOCATION"].astype(str).unique().tolist())

    for seed in SEED_VALUES:
        seed_oof = pd.Series(index=prepared_train.index, dtype=np.float64)
        seed_test = pd.Series(index=prepared_test.index, dtype=np.float64)

        for allocation_value in allocation_values:
            train_alloc_mask = prepared_train["ALLOCATION"].astype(str) == allocation_value
            test_alloc_mask = prepared_test["ALLOCATION"].astype(str) == allocation_value

            train_alloc = prepared_train.loc[train_alloc_mask].copy()
            test_alloc = prepared_test.loc[test_alloc_mask].copy()
            y_alloc = y_train.loc[train_alloc.index, "target"]

            fold_assignments = build_allocation_time_folds(
                train_alloc,
                n_splits=N_SPLITS,
                ts_col="TS",
                fold_col="fold",
            )
            test_fold_scores = []

            for fold_id in range(N_SPLITS):
                train_mask = fold_assignments != fold_id
                valid_mask = fold_assignments == fold_id

                x_local_train = train_alloc.loc[train_mask, feature_cols]
                y_local_train = y_alloc.loc[x_local_train.index]
                x_local_valid = train_alloc.loc[valid_mask, feature_cols]
                y_local_valid = y_alloc.loc[x_local_valid.index]

                model = fit_catboost_regressor(
                    train_x=x_local_train,
                    train_y=y_local_train,
                    random_seed=seed + fold_id,
                )

                valid_scores = model.predict(x_local_valid)
                seed_oof.loc[x_local_valid.index] = valid_scores

                if len(test_alloc) > 0:
                    test_fold_scores.append(model.predict(test_alloc.loc[:, feature_cols]))

                fold_accuracy = directional_accuracy(valid_scores, y_local_valid.to_numpy())
                fold_metrics.append(
                    {
                        "seed": seed,
                        "allocation": allocation_value,
                        "fold": fold_id,
                        "n_train": int(train_mask.sum()),
                        "n_valid": int(valid_mask.sum()),
                        "valid_accuracy": fold_accuracy,
                    }
                )

                model_path = output_dir / f"allocation_{allocation_value}_seed{seed}_fold{fold_id}.cbm"
                model.save_model(str(model_path))
                manifest_rows.append(
                    {
                        "seed": seed,
                        "allocation": allocation_value,
                        "fold": fold_id,
                        "model_path": model_path.name,
                    }
                )
                print(
                    f"[allocation-shape] seed={seed} allocation={allocation_value} fold={fold_id} "
                    f"valid_accuracy={fold_accuracy:.6f}"
                )

            if len(test_alloc) > 0 and test_fold_scores:
                seed_test.loc[test_alloc.index] = np.mean(np.vstack(test_fold_scores), axis=0)

        oof_scores[f"score_seed_{seed}"] = seed_oof.reindex(oof_scores.index)
        test_scores[f"score_seed_{seed}"] = seed_test.reindex(test_scores.index)

    oof_scores["score_mean"] = oof_scores.mean(axis=1)
    test_scores["score_mean"] = test_scores.mean(axis=1)

    oof_output = pd.DataFrame(index=prepared_train.index)
    oof_output["TS"] = prepared_train["TS"].to_numpy()
    oof_output["GROUP"] = prepared_train["GROUP"].to_numpy()
    oof_output["ALLOCATION"] = prepared_train["ALLOCATION"].to_numpy()
    oof_output["target"] = y_train["target"].to_numpy()
    oof_output["target_binary"] = (oof_output["target"].to_numpy() >= 0.0).astype(int)
    for seed in SEED_VALUES:
        oof_output[f"score_seed_{seed}"] = oof_scores[f"score_seed_{seed}"].to_numpy()
    oof_output["score_mean"] = oof_scores["score_mean"].to_numpy()
    oof_output["prediction"] = (oof_output["score_mean"] >= 0.0).astype(int)
    oof_output["ROW_ID"] = oof_output.index

    test_output = pd.DataFrame(index=prepared_test.index)
    test_output["TS"] = prepared_test["TS"].to_numpy()
    test_output["GROUP"] = prepared_test["GROUP"].to_numpy()
    test_output["ALLOCATION"] = prepared_test["ALLOCATION"].to_numpy()
    for seed in SEED_VALUES:
        test_output[f"score_seed_{seed}"] = test_scores[f"score_seed_{seed}"].to_numpy()
    test_output["score_mean"] = test_scores["score_mean"].to_numpy()
    test_output["prediction"] = (test_output["score_mean"] >= 0.0).astype(int)
    test_output["ROW_ID"] = test_output.index

    fold_metrics_df = pd.DataFrame(fold_metrics)
    comparison_df = pd.DataFrame(
        [
            {
                "model": f"catboost_allocation_shape_features_{SEED_VALUES}",
                "oof_accuracy": directional_accuracy(oof_output["score_mean"], oof_output["target"]),
            }
        ]
    )

    manifest = {
        "model_type": "catboost_allocation_shape_features",
        "seed_values": list(SEED_VALUES),
        "n_splits": N_SPLITS,
        "feature_cols": feature_cols,
        "shape_feature_cols": list(SHAPE_FEATURE_COLUMNS),
        "models": manifest_rows,
    }

    oof_output.to_csv(output_dir / "oof_predictions.csv", index=False)
    test_output.to_csv(output_dir / "test_predictions_with_scores.csv", index=False)
    fold_metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    comparison_df.to_csv(output_dir / "comparison.csv", index=False)
    (output_dir / "allocation_shape_features_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    if preds_output_path is not None:
        preds_output_path = Path(preds_output_path)
        preds_output_path.parent.mkdir(parents=True, exist_ok=True)
        test_output[["ROW_ID", "prediction"]].to_csv(preds_output_path, index=False)

    return {
        "oof_output": oof_output,
        "test_output": test_output,
        "fold_metrics_df": fold_metrics_df,
        "comparison_df": comparison_df,
        "manifest": manifest,
    }


def main():
    parser = argparse.ArgumentParser(description="Train allocation-specific CatBoost with extra shape features.")
    parser.add_argument("--x-train", default="data/X_train.csv")
    parser.add_argument("--y-train", default="data/y_train.csv")
    parser.add_argument("--x-test", default="data/X_test.csv")
    parser.add_argument("--output-dir", default="model_catboost_allocation_shape_features")
    parser.add_argument("--preds-output", default="code/preds_catboost_allocation_shape_features.csv")
    args = parser.parse_args()

    result = train_catboost_allocation_shape_features(
        x_train_path=args.x_train,
        y_train_path=args.y_train,
        x_test_path=args.x_test,
        output_dir=args.output_dir,
        preds_output_path=args.preds_output,
    )
    print(result["comparison_df"])


if __name__ == "__main__":
    main()
