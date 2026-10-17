from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MIN_WEIGHT = 0.0
DEFAULT_MAX_WEIGHT = 0.4
DEFAULT_WEIGHT_STEP = 0.05


def directional_accuracy(scores: pd.Series | np.ndarray, targets: pd.Series | np.ndarray) -> float:
    preds = (np.asarray(scores) >= 0.0).astype(int)
    target_binary = (np.asarray(targets) >= 0.0).astype(int)
    return float((preds == target_binary).mean())


def zscore(values: pd.Series, mean: float | None = None, std: float | None = None) -> tuple[pd.Series, float, float]:
    local_mean = float(values.mean()) if mean is None else float(mean)
    local_std = float(values.std()) if std is None else float(std)
    if not np.isfinite(local_std) or local_std <= 1e-12:
        local_std = 1.0
    return (values - local_mean) / local_std, local_mean, local_std


def build_weight_grid(min_weight: float, max_weight: float, step: float) -> tuple[float, ...]:
    n_steps = int(round((max_weight - min_weight) / step))
    weights = [round(min_weight + idx * step, 10) for idx in range(n_steps + 1)]
    if abs(weights[-1] - max_weight) > 1e-10:
        weights.append(round(max_weight, 10))
    return tuple(weights)


def load_frames(
    catboost_model_dir: str | Path,
    gru_model_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    catboost_model_dir = Path(catboost_model_dir)
    gru_model_dir = Path(gru_model_dir)

    cat_oof = pd.read_csv(catboost_model_dir / "oof_predictions.csv").set_index("ROW_ID").sort_index()
    cat_test = pd.read_csv(catboost_model_dir / "test_predictions_with_scores.csv").set_index("ROW_ID").sort_index()
    gru_oof = pd.read_csv(gru_model_dir / "oof_predictions.csv").set_index("ROW_ID").sort_index()
    gru_test = pd.read_csv(gru_model_dir / "test_predictions_with_scores.csv").set_index("ROW_ID").sort_index()

    train_frame = pd.DataFrame(index=cat_oof.index)
    train_frame["GROUP"] = cat_oof["GROUP"].astype(str)
    train_frame["ALLOCATION"] = cat_oof["ALLOCATION"].astype(str)
    train_frame["CLUSTER"] = cat_oof["CLUSTER"].astype(str)
    train_frame["target"] = cat_oof["target"].astype(float)
    train_frame["catboost_score"] = cat_oof["final_score_mean"].astype(float)
    train_frame["gru_score"] = gru_oof.loc[train_frame.index, "final_logit"].astype(float)

    test_frame = pd.DataFrame(index=cat_test.index)
    test_frame["GROUP"] = cat_test["GROUP"].astype(str)
    test_frame["ALLOCATION"] = cat_test["ALLOCATION"].astype(str)
    test_frame["CLUSTER"] = cat_test["CLUSTER"].astype(str)
    test_frame["catboost_score"] = cat_test["final_score_mean"].astype(float)
    test_frame["gru_score"] = gru_test.loc[test_frame.index, "final_logit"].astype(float)

    train_cat_z, cat_mean, cat_std = zscore(train_frame["catboost_score"])
    train_gru_z, gru_mean, gru_std = zscore(train_frame["gru_score"])
    test_cat_z, _, _ = zscore(test_frame["catboost_score"], mean=cat_mean, std=cat_std)
    test_gru_z, _, _ = zscore(test_frame["gru_score"], mean=gru_mean, std=gru_std)

    train_frame["catboost_z"] = train_cat_z.to_numpy()
    train_frame["gru_z"] = train_gru_z.to_numpy()
    test_frame["catboost_z"] = test_cat_z.to_numpy()
    test_frame["gru_z"] = test_gru_z.to_numpy()

    return train_frame, test_frame


def blended_score(cat_scores: pd.Series, gru_scores: pd.Series, gru_weight: float) -> pd.Series:
    return (1.0 - gru_weight) * cat_scores + gru_weight * gru_scores


def search_segmented_weights(
    train_frame: pd.DataFrame,
    segment_col: str,
    weights: tuple[float, ...],
) -> tuple[dict[str, float], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    best_weights: dict[str, float] = {}

    for segment_value, segment_df in train_frame.groupby(segment_col, sort=True):
        best_weight = None
        best_accuracy = None
        for weight in weights:
            score = blended_score(segment_df["catboost_z"], segment_df["gru_z"], weight)
            accuracy = directional_accuracy(score, segment_df["target"])
            rows.append(
                {
                    "mode": segment_col.lower(),
                    "segment": str(segment_value),
                    "gru_weight": float(weight),
                    "oof_accuracy": accuracy,
                    "n_rows": int(len(segment_df)),
                }
            )
            if best_accuracy is None or accuracy > best_accuracy or (accuracy == best_accuracy and float(weight) < float(best_weight)):
                best_accuracy = accuracy
                best_weight = weight
        best_weights[str(segment_value)] = float(best_weight)

    search_df = pd.DataFrame(rows).sort_values(["segment", "oof_accuracy", "gru_weight"], ascending=[True, False, True]).reset_index(drop=True)
    return best_weights, search_df


def apply_segmented_blend(frame: pd.DataFrame, segment_col: str, weights_by_segment: dict[str, float]) -> pd.Series:
    blend = frame["catboost_z"].copy()
    for segment_value, segment_weight in weights_by_segment.items():
        mask = frame[segment_col].astype(str) == str(segment_value)
        if mask.any():
            blend.loc[mask] = blended_score(
                frame.loc[mask, "catboost_z"],
                frame.loc[mask, "gru_z"],
                segment_weight,
            )
    return blend


def save_submission(test_frame: pd.DataFrame, score: pd.Series, output_path: Path) -> None:
    submission = pd.DataFrame(
        {
            "ROW_ID": test_frame.index,
            "prediction": (score >= 0.0).astype(int),
        }
    )
    submission.to_csv(output_path, index=False)


def run_blend(
    catboost_model_dir: str | Path,
    gru_model_dir: str | Path,
    output_dir: str | Path,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    weight_step: float = DEFAULT_WEIGHT_STEP,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = build_weight_grid(min_weight=min_weight, max_weight=max_weight, step=weight_step)
    train_frame, test_frame = load_frames(
        catboost_model_dir=catboost_model_dir,
        gru_model_dir=gru_model_dir,
    )

    cat_accuracy = directional_accuracy(train_frame["catboost_z"], train_frame["target"])
    gru_accuracy = directional_accuracy(train_frame["gru_z"], train_frame["target"])

    cluster_weights, cluster_search_df = search_segmented_weights(train_frame, "CLUSTER", weights)
    cluster_oof_score = apply_segmented_blend(train_frame, "CLUSTER", cluster_weights)
    cluster_test_score = apply_segmented_blend(test_frame, "CLUSTER", cluster_weights)

    allocation_weights, allocation_search_df = search_segmented_weights(train_frame, "ALLOCATION", weights)
    allocation_oof_score = apply_segmented_blend(train_frame, "ALLOCATION", allocation_weights)
    allocation_test_score = apply_segmented_blend(test_frame, "ALLOCATION", allocation_weights)

    oof_output = train_frame.loc[:, ["GROUP", "ALLOCATION", "CLUSTER", "target", "catboost_score", "gru_score", "catboost_z", "gru_z"]].copy()
    oof_output["cluster_blend_score"] = cluster_oof_score.to_numpy()
    oof_output["allocation_blend_score"] = allocation_oof_score.to_numpy()
    oof_output["cluster_prediction"] = (oof_output["cluster_blend_score"] >= 0.0).astype(int)
    oof_output["allocation_prediction"] = (oof_output["allocation_blend_score"] >= 0.0).astype(int)
    oof_output["ROW_ID"] = oof_output.index

    test_output = test_frame.loc[:, ["GROUP", "ALLOCATION", "CLUSTER", "catboost_score", "gru_score", "catboost_z", "gru_z"]].copy()
    test_output["cluster_blend_score"] = cluster_test_score.to_numpy()
    test_output["allocation_blend_score"] = allocation_test_score.to_numpy()
    test_output["cluster_prediction"] = (test_output["cluster_blend_score"] >= 0.0).astype(int)
    test_output["allocation_prediction"] = (test_output["allocation_blend_score"] >= 0.0).astype(int)
    test_output["ROW_ID"] = test_output.index

    comparison_df = pd.DataFrame(
        [
            {"model": "catboost_cluster_allocation_residual_z", "oof_accuracy": cat_accuracy},
            {"model": "gru_global_cluster_residual_z", "oof_accuracy": gru_accuracy},
            {"model": "cluster_aware_linear_blend", "oof_accuracy": directional_accuracy(cluster_oof_score, train_frame["target"])},
            {"model": "allocation_aware_linear_blend", "oof_accuracy": directional_accuracy(allocation_oof_score, train_frame["target"])},
        ]
    )

    weights_df = pd.DataFrame(
        [{"mode": "cluster", "segment": key, "gru_weight": value} for key, value in sorted(cluster_weights.items())]
        + [{"mode": "allocation", "segment": key, "gru_weight": value} for key, value in sorted(allocation_weights.items())]
    )

    manifest = {
        "catboost_model_dir": str(Path(catboost_model_dir)),
        "gru_model_dir": str(Path(gru_model_dir)),
        "weight_grid": list(weights),
        "cluster_weights": cluster_weights,
        "allocation_weights": allocation_weights,
    }

    oof_output.to_csv(output_dir / "oof_predictions.csv", index=False)
    test_output.to_csv(output_dir / "test_predictions_with_scores.csv", index=False)
    comparison_df.to_csv(output_dir / "comparison.csv", index=False)
    weights_df.to_csv(output_dir / "selected_weights.csv", index=False)
    cluster_search_df.to_csv(output_dir / "cluster_search.csv", index=False)
    allocation_search_df.to_csv(output_dir / "allocation_search.csv", index=False)
    (output_dir / "blend_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    save_submission(test_frame, cluster_test_score, output_dir / "preds_cluster_aware_linear_blend.csv")
    save_submission(test_frame, allocation_test_score, output_dir / "preds_allocation_aware_linear_blend.csv")

    return {
        "oof_output": oof_output,
        "test_output": test_output,
        "comparison_df": comparison_df,
        "weights_df": weights_df,
        "manifest": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend CatBoost cluster-allocation residual with GRU global-cluster residual.")
    parser.add_argument("--catboost-model-dir", default="model_catboost_cluster_allocation_residual")
    parser.add_argument("--gru-model-dir", default="model_gru_global_cluster_residual")
    parser.add_argument("--output-dir", default="model_catboost_gru_segmented_linear_blend")
    parser.add_argument("--min-weight", type=float, default=DEFAULT_MIN_WEIGHT)
    parser.add_argument("--max-weight", type=float, default=DEFAULT_MAX_WEIGHT)
    parser.add_argument("--weight-step", type=float, default=DEFAULT_WEIGHT_STEP)
    args = parser.parse_args()

    result = run_blend(
        catboost_model_dir=args.catboost_model_dir,
        gru_model_dir=args.gru_model_dir,
        output_dir=args.output_dir,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        weight_step=args.weight_step,
    )
    print(result["comparison_df"])


if __name__ == "__main__":
    main()
