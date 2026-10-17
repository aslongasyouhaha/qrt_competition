from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from allocation_style_clustering_eda import (
    build_allocation_style_frame,
    get_style_feature_columns,
)
from catboost_allocation_shape_features_training import (
    CATBOOST_PARAMS,
    SHAPE_FEATURE_COLUMNS,
    build_shape_features,
)
from utils import (
    apply_standard_scaler,
    build_benchmark_features,
    fit_standard_scaler,
    get_catboost_global_feature_columns,
    seed_everything,
)


SEED_VALUES = (42, 52, 62)
N_SPLITS = 5
N_CLUSTERS = 5
CLUSTER_RANDOM_STATE = 42
FILLNA_VALUE = 0.0
SCALE_FEATURES = True


def directional_accuracy(scores, targets) -> float:
    preds = (np.asarray(scores) >= 0.0).astype(int)
    target_binary = (np.asarray(targets) >= 0.0).astype(int)
    return float((preds == target_binary).mean())


def build_time_folds(df: pd.DataFrame, n_splits: int = 5, ts_col: str = "TS") -> pd.Series:
    unique_ts = sorted(df[ts_col].dropna().unique().tolist())
    if len(unique_ts) < n_splits:
        raise ValueError(f"Not enough unique TS to build {n_splits} folds: only {len(unique_ts)}")
    fold_map = {ts_value: idx % n_splits for idx, ts_value in enumerate(unique_ts)}
    return df[ts_col].map(fold_map).astype(int)


def build_allocation_cluster_map(
    x_train_path: str | Path,
    n_clusters: int = N_CLUSTERS,
    random_state: int = CLUSTER_RANDOM_STATE,
) -> tuple[dict[str, int], pd.DataFrame]:
    style_frame = build_allocation_style_frame(x_train_path)
    feature_cols = get_style_feature_columns(style_frame)
    style_features = style_frame.loc[:, feature_cols].fillna(0.0)
    scaled = StandardScaler().fit_transform(style_features)
    cluster_labels = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20,
    ).fit_predict(scaled)

    cluster_frame = style_frame.loc[:, ["ALLOCATION", "GROUP"]].copy()
    cluster_frame["CLUSTER"] = cluster_labels.astype(int)
    allocation_to_cluster = dict(zip(cluster_frame["ALLOCATION"], cluster_frame["CLUSTER"]))
    return allocation_to_cluster, cluster_frame


def prepare_frames(
    x_train_path: str | Path,
    y_train_path: str | Path,
    x_test_path: str | Path,
    allocation_to_cluster: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    x_train = pd.read_csv(x_train_path, index_col="ROW_ID")
    y_train = pd.read_csv(y_train_path, index_col="ROW_ID")
    x_test = pd.read_csv(x_test_path, index_col="ROW_ID")

    x_train = build_shape_features(build_benchmark_features(x_train))
    x_test = build_shape_features(build_benchmark_features(x_test))

    x_train["ALLOCATION"] = x_train["ALLOCATION"].astype(str)
    x_test["ALLOCATION"] = x_test["ALLOCATION"].astype(str)
    x_train["CLUSTER"] = x_train["ALLOCATION"].map(allocation_to_cluster).astype(int).astype(str)
    x_test["CLUSTER"] = x_test["ALLOCATION"].map(allocation_to_cluster).astype(int).astype(str)

    numeric_feature_cols, _ = get_catboost_global_feature_columns(x_train, include_group=True)
    feature_cols = list(dict.fromkeys([*numeric_feature_cols, *SHAPE_FEATURE_COLUMNS]))

    scaler_frame = x_train.copy()
    scaler_frame.loc[:, feature_cols] = scaler_frame.loc[:, feature_cols].fillna(FILLNA_VALUE)
    scaler = fit_standard_scaler(scaler_frame, feature_cols)

    prepared_train = x_train.copy()
    prepared_test = x_test.copy()
    prepared_train.loc[:, feature_cols] = prepared_train.loc[:, feature_cols].fillna(FILLNA_VALUE)
    prepared_test.loc[:, feature_cols] = prepared_test.loc[:, feature_cols].fillna(FILLNA_VALUE)

    if SCALE_FEATURES:
        prepared_train.loc[:, feature_cols] = scaler.transform(prepared_train.loc[:, feature_cols]).astype(np.float32)
        prepared_test.loc[:, feature_cols] = scaler.transform(prepared_test.loc[:, feature_cols]).astype(np.float32)

    prepared_train["GROUP"] = prepared_train["GROUP"].astype(str)
    prepared_test["GROUP"] = prepared_test["GROUP"].astype(str)
    return prepared_train, y_train, prepared_test, feature_cols


def fit_catboost_regressor(train_x: pd.DataFrame, train_y: pd.Series, random_seed: int) -> CatBoostRegressor:
    model = CatBoostRegressor(**{**CATBOOST_PARAMS, "random_seed": random_seed})
    model.fit(train_x, train_y)
    return model


def train_catboost_cluster_allocation_residual(
    x_train_path: str | Path,
    y_train_path: str | Path,
    x_test_path: str | Path,
    output_dir: str | Path,
    preds_output_path: str | Path | None = None,
    n_clusters: int = N_CLUSTERS,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED_VALUES[0])

    allocation_to_cluster, cluster_frame = build_allocation_cluster_map(
        x_train_path=x_train_path,
        n_clusters=n_clusters,
        random_state=CLUSTER_RANDOM_STATE,
    )
    prepared_train, y_train, prepared_test, feature_cols = prepare_frames(
        x_train_path=x_train_path,
        y_train_path=y_train_path,
        x_test_path=x_test_path,
        allocation_to_cluster=allocation_to_cluster,
    )

    oof_scores = pd.DataFrame(index=prepared_train.index)
    test_scores = pd.DataFrame(index=prepared_test.index)
    fold_metrics: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []

    cluster_values = sorted(prepared_train["CLUSTER"].unique().tolist())
    allocation_values = sorted(prepared_train["ALLOCATION"].unique().tolist())

    for seed in SEED_VALUES:
        seed_base_oof = pd.Series(index=prepared_train.index, dtype=np.float64)
        seed_base_test = pd.Series(index=prepared_test.index, dtype=np.float64)

        for cluster_value in tqdm(
            cluster_values,
            desc=f"[base] seed={seed} clusters",
            leave=True,
        ):
            train_cluster_mask = prepared_train["CLUSTER"] == cluster_value
            test_cluster_mask = prepared_test["CLUSTER"] == cluster_value

            train_cluster = prepared_train.loc[train_cluster_mask].copy()
            test_cluster = prepared_test.loc[test_cluster_mask].copy()
            y_cluster = y_train.loc[train_cluster.index, "target"]

            fold_assignments = build_time_folds(train_cluster, n_splits=N_SPLITS, ts_col="TS")
            test_fold_scores = []

            for fold_id in tqdm(
                range(N_SPLITS),
                desc=f"  cluster={cluster_value} folds",
                leave=False,
            ):
                train_mask = fold_assignments != fold_id
                valid_mask = fold_assignments == fold_id

                x_local_train = train_cluster.loc[train_mask, feature_cols]
                y_local_train = y_cluster.loc[x_local_train.index]
                x_local_valid = train_cluster.loc[valid_mask, feature_cols]
                y_local_valid = y_cluster.loc[x_local_valid.index]

                model = fit_catboost_regressor(
                    train_x=x_local_train,
                    train_y=y_local_train,
                    random_seed=seed + fold_id,
                )
                valid_scores = model.predict(x_local_valid)
                seed_base_oof.loc[x_local_valid.index] = valid_scores

                if len(test_cluster) > 0:
                    test_fold_scores.append(model.predict(test_cluster.loc[:, feature_cols]))

                fold_metrics.append(
                    {
                        "stage": "cluster_base",
                        "seed": seed,
                        "cluster": cluster_value,
                        "fold": fold_id,
                        "n_train": int(train_mask.sum()),
                        "n_valid": int(valid_mask.sum()),
                        "valid_accuracy": directional_accuracy(valid_scores, y_local_valid.to_numpy()),
                    }
                )

                model_path = output_dir / f"cluster_{cluster_value}_base_seed{seed}_fold{fold_id}.cbm"
                model.save_model(str(model_path))
                manifest_rows.append(
                    {
                        "stage": "cluster_base",
                        "seed": seed,
                        "cluster": cluster_value,
                        "fold": fold_id,
                        "model_path": model_path.name,
                    }
                )

            if len(test_cluster) > 0 and test_fold_scores:
                seed_base_test.loc[test_cluster.index] = np.mean(np.vstack(test_fold_scores), axis=0)

        if seed_base_oof.isna().any():
            raise RuntimeError(f"Missing base OOF scores for seed {seed}.")

        residual_targets = y_train["target"] - seed_base_oof
        seed_resid_oof = pd.Series(index=prepared_train.index, dtype=np.float64)
        seed_resid_test = pd.Series(index=prepared_test.index, dtype=np.float64)

        for allocation_value in tqdm(
            allocation_values,
            desc=f"[residual] seed={seed} allocations",
            leave=True,
        ):
            train_alloc_mask = prepared_train["ALLOCATION"] == allocation_value
            test_alloc_mask = prepared_test["ALLOCATION"] == allocation_value

            train_alloc = prepared_train.loc[train_alloc_mask].copy()
            test_alloc = prepared_test.loc[test_alloc_mask].copy()
            y_alloc = residual_targets.loc[train_alloc.index]

            fold_assignments = build_time_folds(train_alloc, n_splits=N_SPLITS, ts_col="TS")
            test_fold_scores = []

            for fold_id in tqdm(
                range(N_SPLITS),
                desc=f"  allocation={allocation_value} folds",
                leave=False,
            ):
                train_mask = fold_assignments != fold_id
                valid_mask = fold_assignments == fold_id

                x_local_train = train_alloc.loc[train_mask, feature_cols]
                y_local_train = y_alloc.loc[x_local_train.index]
                x_local_valid = train_alloc.loc[valid_mask, feature_cols]
                y_local_valid = y_alloc.loc[x_local_valid.index]

                model = fit_catboost_regressor(
                    train_x=x_local_train,
                    train_y=y_local_train,
                    random_seed=seed + 100 + fold_id,
                )
                valid_scores = model.predict(x_local_valid)
                seed_resid_oof.loc[x_local_valid.index] = valid_scores

                if len(test_alloc) > 0:
                    test_fold_scores.append(model.predict(test_alloc.loc[:, feature_cols]))

                fold_metrics.append(
                    {
                        "stage": "allocation_residual",
                        "seed": seed,
                        "allocation": allocation_value,
                        "fold": fold_id,
                        "n_train": int(train_mask.sum()),
                        "n_valid": int(valid_mask.sum()),
                        "valid_accuracy": directional_accuracy(
                            seed_base_oof.loc[x_local_valid.index].to_numpy() + valid_scores,
                            y_train.loc[x_local_valid.index, "target"].to_numpy(),
                        ),
                    }
                )

                model_path = output_dir / f"allocation_{allocation_value}_residual_seed{seed}_fold{fold_id}.cbm"
                model.save_model(str(model_path))
                manifest_rows.append(
                    {
                        "stage": "allocation_residual",
                        "seed": seed,
                        "allocation": allocation_value,
                        "fold": fold_id,
                        "model_path": model_path.name,
                    }
                )

            if len(test_alloc) > 0 and test_fold_scores:
                seed_resid_test.loc[test_alloc.index] = np.mean(np.vstack(test_fold_scores), axis=0)

        if seed_resid_oof.isna().any():
            raise RuntimeError(f"Missing residual OOF scores for seed {seed}.")

        oof_scores[f"base_score_seed_{seed}"] = seed_base_oof.reindex(oof_scores.index)
        oof_scores[f"residual_score_seed_{seed}"] = seed_resid_oof.reindex(oof_scores.index)
        oof_scores[f"final_score_seed_{seed}"] = oof_scores[f"base_score_seed_{seed}"] + oof_scores[f"residual_score_seed_{seed}"]

        test_scores[f"base_score_seed_{seed}"] = seed_base_test.reindex(test_scores.index)
        test_scores[f"residual_score_seed_{seed}"] = seed_resid_test.reindex(test_scores.index)
        test_scores[f"final_score_seed_{seed}"] = test_scores[f"base_score_seed_{seed}"] + test_scores[f"residual_score_seed_{seed}"]

    final_seed_cols = [f"final_score_seed_{seed}" for seed in SEED_VALUES]
    base_seed_cols = [f"base_score_seed_{seed}" for seed in SEED_VALUES]
    resid_seed_cols = [f"residual_score_seed_{seed}" for seed in SEED_VALUES]
    oof_scores["base_score_mean"] = oof_scores.loc[:, base_seed_cols].mean(axis=1)
    oof_scores["residual_score_mean"] = oof_scores.loc[:, resid_seed_cols].mean(axis=1)
    oof_scores["final_score_mean"] = oof_scores.loc[:, final_seed_cols].mean(axis=1)
    test_scores["base_score_mean"] = test_scores.loc[:, base_seed_cols].mean(axis=1)
    test_scores["residual_score_mean"] = test_scores.loc[:, resid_seed_cols].mean(axis=1)
    test_scores["final_score_mean"] = test_scores.loc[:, final_seed_cols].mean(axis=1)

    oof_output = pd.DataFrame(index=prepared_train.index)
    oof_output["TS"] = prepared_train["TS"].to_numpy()
    oof_output["GROUP"] = prepared_train["GROUP"].to_numpy()
    oof_output["CLUSTER"] = prepared_train["CLUSTER"].to_numpy()
    oof_output["ALLOCATION"] = prepared_train["ALLOCATION"].to_numpy()
    oof_output["target"] = y_train["target"].to_numpy()
    oof_output["target_binary"] = (oof_output["target"].to_numpy() >= 0.0).astype(int)
    for col in [*base_seed_cols, *resid_seed_cols, *final_seed_cols, "base_score_mean", "residual_score_mean", "final_score_mean"]:
        oof_output[col] = oof_scores[col].to_numpy()
    oof_output["prediction"] = (oof_output["final_score_mean"] >= 0.0).astype(int)
    oof_output["ROW_ID"] = oof_output.index

    test_output = pd.DataFrame(index=prepared_test.index)
    test_output["TS"] = prepared_test["TS"].to_numpy()
    test_output["GROUP"] = prepared_test["GROUP"].to_numpy()
    test_output["CLUSTER"] = prepared_test["CLUSTER"].to_numpy()
    test_output["ALLOCATION"] = prepared_test["ALLOCATION"].to_numpy()
    for col in [*base_seed_cols, *resid_seed_cols, *final_seed_cols, "base_score_mean", "residual_score_mean", "final_score_mean"]:
        test_output[col] = test_scores[col].to_numpy()
    test_output["prediction"] = (test_output["final_score_mean"] >= 0.0).astype(int)
    test_output["ROW_ID"] = test_output.index

    fold_metrics_df = pd.DataFrame(fold_metrics)
    comparison_df = pd.DataFrame(
        [
            {
                "model": "catboost_cluster_allocation_residual",
                "n_clusters": n_clusters,
                "oof_accuracy": directional_accuracy(oof_output["final_score_mean"], oof_output["target"]),
                "base_oof_accuracy": directional_accuracy(oof_output["base_score_mean"], oof_output["target"]),
            }
        ]
    )

    manifest = {
        "model_type": "catboost_cluster_allocation_residual",
        "seed_values": list(SEED_VALUES),
        "n_splits": N_SPLITS,
        "n_clusters": n_clusters,
        "feature_cols": feature_cols,
        "allocation_cluster_map": cluster_frame.to_dict(orient="records"),
        "models": manifest_rows,
    }

    oof_output.to_csv(output_dir / "oof_predictions.csv", index=False)
    test_output.to_csv(output_dir / "test_predictions_with_scores.csv", index=False)
    fold_metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    comparison_df.to_csv(output_dir / "comparison.csv", index=False)
    (output_dir / "cluster_allocation_residual_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

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
        "cluster_frame": cluster_frame,
    }


def main():
    parser = argparse.ArgumentParser(description="Train CatBoost with shape features using cluster base + allocation residual.")
    parser.add_argument("--x-train", default="data/X_train.csv")
    parser.add_argument("--y-train", default="data/y_train.csv")
    parser.add_argument("--x-test", default="data/X_test.csv")
    parser.add_argument("--output-dir", default="model_catboost_cluster_allocation_residual")
    parser.add_argument("--preds-output", default="code/preds_catboost_cluster_allocation_residual.csv")
    parser.add_argument("--n-clusters", default=N_CLUSTERS, type=int)
    args = parser.parse_args()

    result = train_catboost_cluster_allocation_residual(
        x_train_path=args.x_train,
        y_train_path=args.y_train,
        x_test_path=args.x_test,
        output_dir=args.output_dir,
        preds_output_path=args.preds_output,
        n_clusters=args.n_clusters,
    )
    print(result["comparison_df"])


if __name__ == "__main__":
    main()
