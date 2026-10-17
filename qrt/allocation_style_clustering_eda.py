from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


RET_COLS = [f"RET_{i}" for i in range(1, 21)]
VOL_COLS = [f"SIGNED_VOLUME_{i}" for i in range(1, 21)]
RAW_USECOLS = ["ALLOCATION", "GROUP", "MEDIAN_DAILY_TURNOVER", *RET_COLS, *VOL_COLS]


@dataclass
class AllocationClusteringConfig:
    n_clusters: int = 8
    random_state: int = 42
    pca_components: int = 2
    figure_dpi: int = 180


def build_allocation_style_frame(x_train_path: str | Path) -> pd.DataFrame:
    x_train = pd.read_csv(x_train_path, usecols=RAW_USECOLS)
    x_train["ALLOCATION"] = x_train["ALLOCATION"].astype(str)

    agg_spec: dict[str, list[str]] = {"GROUP": ["first"], "MEDIAN_DAILY_TURNOVER": ["mean", "std"]}
    for col in [*RET_COLS, *VOL_COLS]:
        agg_spec[col] = ["mean", "std"]

    grouped = x_train.groupby("ALLOCATION", sort=True).agg(agg_spec)
    grouped.columns = [
        "GROUP" if col == "GROUP" and stat == "first" else f"{col}_{stat.upper()}"
        for col, stat in grouped.columns
    ]
    grouped = grouped.reset_index()

    ret_missing = (
        x_train.groupby("ALLOCATION")[RET_COLS]
        .apply(lambda frame: frame.isna().mean().mean())
        .rename("RET_MISSING_RATE")
        .reset_index()
    )
    vol_missing = (
        x_train.groupby("ALLOCATION")[VOL_COLS]
        .apply(lambda frame: frame.isna().mean().mean())
        .rename("VOL_MISSING_RATE")
        .reset_index()
    )

    style_frame = grouped.merge(ret_missing, on="ALLOCATION", how="left")
    style_frame = style_frame.merge(vol_missing, on="ALLOCATION", how="left")
    return style_frame


def get_style_feature_columns(style_frame: pd.DataFrame) -> list[str]:
    return [
        col
        for col in style_frame.columns
        if col not in {"ALLOCATION", "GROUP"}
    ]


def run_allocation_style_clustering(
    x_train_path: str | Path,
    output_dir: str | Path,
    config: AllocationClusteringConfig | None = None,
):
    config = config or AllocationClusteringConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    style_frame = build_allocation_style_frame(x_train_path)
    feature_cols = get_style_feature_columns(style_frame)

    feature_matrix = style_frame.loc[:, feature_cols].copy().fillna(0.0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)

    pca = PCA(n_components=config.pca_components, random_state=config.random_state)
    coords = pca.fit_transform(scaled)

    clusterer = KMeans(
        n_clusters=config.n_clusters,
        random_state=config.random_state,
        n_init=20,
    )
    cluster_labels = clusterer.fit_predict(scaled)

    result_frame = style_frame.loc[:, ["ALLOCATION", "GROUP"]].copy()
    result_frame["cluster"] = cluster_labels
    result_frame["pca_1"] = coords[:, 0]
    result_frame["pca_2"] = coords[:, 1]

    explained_var = pd.DataFrame(
        {
            "component": [f"PC{i+1}" for i in range(config.pca_components)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )

    cluster_group_counts = (
        result_frame.groupby(["cluster", "GROUP"])
        .size()
        .unstack(fill_value=0)
        .sort_index(axis=1)
    )
    cluster_group_ratio = cluster_group_counts.div(cluster_group_counts.sum(axis=1), axis=0)

    cluster_summary = []
    for cluster_id, cluster_rows in result_frame.groupby("cluster"):
        group_counts = cluster_rows["GROUP"].value_counts().sort_index()
        dominant_group = int(group_counts.idxmax())
        cross_group_count = int((group_counts > 0).sum())
        cluster_summary.append(
            {
                "cluster": int(cluster_id),
                "n_allocations": int(len(cluster_rows)),
                "dominant_group": dominant_group,
                "cross_group_count": cross_group_count,
                "is_cross_group_cluster": bool(cross_group_count >= 2),
            }
        )
    cluster_summary_df = pd.DataFrame(cluster_summary).sort_values("cluster")

    cross_group_members = (
        result_frame.merge(cluster_summary_df.loc[:, ["cluster", "is_cross_group_cluster"]], on="cluster", how="left")
        .query("is_cross_group_cluster")
        .sort_values(["cluster", "GROUP", "ALLOCATION"])
    )

    result_frame.to_csv(output_dir / "allocation_style_points.csv", index=False)
    cluster_group_counts.to_csv(output_dir / "cluster_group_counts.csv")
    cluster_group_ratio.to_csv(output_dir / "cluster_group_ratio.csv")
    cluster_summary_df.to_csv(output_dir / "cluster_summary.csv", index=False)
    cross_group_members.to_csv(output_dir / "cross_group_cluster_members.csv", index=False)
    explained_var.to_csv(output_dir / "pca_explained_variance.csv", index=False)

    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=result_frame,
        x="pca_1",
        y="pca_2",
        hue="GROUP",
        style="cluster",
        palette="tab10",
        s=70,
        alpha=0.9,
    )
    plt.title("Allocation Style Clusters: PCA colored by GROUP")
    plt.tight_layout()
    plt.savefig(output_dir / "allocation_style_pca_by_group.png", dpi=config.figure_dpi)
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=result_frame,
        x="pca_1",
        y="pca_2",
        hue="cluster",
        style="GROUP",
        palette="tab10",
        s=70,
        alpha=0.9,
    )
    plt.title("Allocation Style Clusters: PCA colored by cluster")
    plt.tight_layout()
    plt.savefig(output_dir / "allocation_style_pca_by_cluster.png", dpi=config.figure_dpi)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.heatmap(cluster_group_ratio, annot=True, cmap="Blues", fmt=".2f")
    plt.title("Cluster x GROUP ratio")
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_group_ratio_heatmap.png", dpi=config.figure_dpi)
    plt.close()

    return {
        "style_frame": style_frame,
        "result_frame": result_frame,
        "cluster_group_counts": cluster_group_counts,
        "cluster_group_ratio": cluster_group_ratio,
        "cluster_summary_df": cluster_summary_df,
        "cross_group_members": cross_group_members,
        "explained_var": explained_var,
        "feature_cols": feature_cols,
    }
