from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from allocation_style_clustering_eda import build_allocation_style_frame, get_style_feature_columns
from gru_global_training import (
    GLOBAL_SEQUENCE_FEATURE_TEMPLATES,
    GROUP_SEQUENCE_FEATURE_TEMPLATES,
    STATIC_FEATURE_COLS,
    GRUGlobalConfig,
    build_gru_timeseries_features,
    fit_and_save_global_gru,
    get_gru_global_feature_columns,
    predict_global_gru,
)
from utils import fit_standard_scaler, seed_everything


SEED_VALUES = (42, 52, 62)
N_SPLITS = 5
N_CLUSTERS = 5
CLUSTER_RANDOM_STATE = 42


@dataclass
class GRUGlobalClusterResidualConfig:
    cluster_col: str = "CLUSTER"
    ts_col: str = "TS"
    target_col: str = "target"
    n_splits: int = 5
    global_input_size: int = len(GLOBAL_SEQUENCE_FEATURE_TEMPLATES)
    group_input_size: int = len(GROUP_SEQUENCE_FEATURE_TEMPLATES)
    static_input_size: int = len(STATIC_FEATURE_COLS)
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 16
    patience: int = 3
    grad_clip_norm: float = 1.0
    fillna_value: float = 0.0
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    scale_features: bool = True
    decision_threshold: float = 0.5
    residual_mse_weight: float = 0.25


def _resolve_device(device: str | object) -> str:
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        return device
    return "cpu"


def build_allocation_cluster_map(
    x_train_path: str | Path,
    n_clusters: int = N_CLUSTERS,
    random_state: int = CLUSTER_RANDOM_STATE,
) -> tuple[dict[str, int], pd.DataFrame]:
    style_frame = build_allocation_style_frame(x_train_path)
    feature_cols = get_style_feature_columns(style_frame)
    scaled = fit_standard_scaler(style_frame.loc[:, feature_cols].fillna(0.0), feature_cols).transform(
        style_frame.loc[:, feature_cols].fillna(0.0)
    )
    from sklearn.cluster import KMeans

    cluster_labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20).fit_predict(scaled)
    cluster_frame = style_frame.loc[:, ["ALLOCATION", "GROUP"]].copy()
    cluster_frame["CLUSTER"] = cluster_labels.astype(int)
    allocation_to_cluster = dict(zip(cluster_frame["ALLOCATION"].astype(str), cluster_frame["CLUSTER"]))
    return allocation_to_cluster, cluster_frame


def build_time_folds(df: pd.DataFrame, n_splits: int = 5, ts_col: str = "TS") -> pd.Series:
    unique_ts = sorted(df[ts_col].dropna().unique().tolist())
    if len(unique_ts) < n_splits:
        raise ValueError(f"Not enough unique TS to build {n_splits} folds: only {len(unique_ts)}")
    fold_map = {ts_value: idx % n_splits for idx, ts_value in enumerate(unique_ts)}
    return df[ts_col].map(fold_map).astype(int)


def prepare_dataframe(
    df: pd.DataFrame,
    feature_cols: list[str],
    config: GRUGlobalClusterResidualConfig,
    scaler=None,
    fit_scaler: bool = False,
):
    prepared = df.copy()
    prepared.loc[:, feature_cols] = prepared.loc[:, feature_cols].fillna(config.fillna_value)
    fitted_scaler = scaler
    if config.scale_features:
        if fit_scaler:
            fitted_scaler = fit_standard_scaler(prepared, feature_cols)
        elif scaler is None:
            raise ValueError("A scaler must be provided when fit_scaler is False.")
        scaled = fitted_scaler.transform(prepared.loc[:, feature_cols]).astype(np.float32, copy=False)
        for col_idx, col_name in enumerate(feature_cols):
            prepared[col_name] = scaled[:, col_idx]
    return prepared, fitted_scaler


def dataframe_to_inputs(df: pd.DataFrame):
    global_grouped = [[template.format(lag=lag) for template in GLOBAL_SEQUENCE_FEATURE_TEMPLATES] for lag in range(20, 0, -1)]
    group_grouped = [[template.format(lag=lag) for template in GROUP_SEQUENCE_FEATURE_TEMPLATES] for lag in range(20, 0, -1)]
    global_sequences = np.stack([df.loc[:, cols].to_numpy(dtype=np.float32) for cols in global_grouped], axis=1)
    group_sequences = np.stack([df.loc[:, cols].to_numpy(dtype=np.float32) for cols in group_grouped], axis=1)
    static_features = df.loc[:, list(STATIC_FEATURE_COLS)].to_numpy(dtype=np.float32)
    return global_sequences, group_sequences, static_features


class ClusterResidualDataset(Dataset):
    def __init__(self, global_sequences, group_sequences, static_features, base_logits, targets) -> None:
        self.global_sequences = torch.tensor(global_sequences, dtype=torch.float32)
        self.group_sequences = torch.tensor(group_sequences, dtype=torch.float32)
        self.static_features = torch.tensor(static_features, dtype=torch.float32)
        self.base_logits = torch.tensor(base_logits, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            self.global_sequences[idx],
            self.group_sequences[idx],
            self.static_features[idx],
            self.base_logits[idx],
            self.targets[idx],
        )


def build_dataloader(
    global_sequences,
    group_sequences,
    static_features,
    base_logits,
    targets,
    config: GRUGlobalClusterResidualConfig,
    shuffle: bool,
):
    dataset = ClusterResidualDataset(global_sequences, group_sequences, static_features, base_logits, targets)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
    )


class ClusterResidualGRU(nn.Module):
    def __init__(self, config: GRUGlobalClusterResidualConfig) -> None:
        super().__init__()
        gru_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.global_gru = nn.GRU(
            input_size=config.global_input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.group_gru = nn.GRU(
            input_size=config.group_input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.global_static_proj = nn.Linear(config.static_input_size, config.hidden_size)
        self.group_static_proj = nn.Linear(config.static_input_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.global_fuse = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.group_fuse = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size * 2),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1),
        )

    def forward(self, global_sequences, group_sequences, static_features):
        global_output, _ = self.global_gru(global_sequences)
        group_output, _ = self.group_gru(group_sequences)
        global_repr = self.global_fuse(self.dropout(global_output[:, -1, :]) + self.global_static_proj(static_features))
        group_repr = self.group_fuse(self.dropout(group_output[:, -1, :]) + self.group_static_proj(static_features))
        fused = torch.cat([global_repr, group_repr], dim=1)
        return self.residual_head(fused).squeeze(-1)


def train_one_epoch(model, dataloader, criterion, optimizer, config: GRUGlobalClusterResidualConfig):
    model.train()
    total_loss = total_bce = total_mse = 0.0
    total_correct = total_count = 0
    for global_seq, group_seq, static_x, base_logits, targets in dataloader:
        global_seq = global_seq.to(config.device)
        group_seq = group_seq.to(config.device)
        static_x = static_x.to(config.device)
        base_logits = base_logits.to(config.device)
        targets = targets.to(config.device)

        optimizer.zero_grad(set_to_none=True)
        residual_logits = model(global_seq, group_seq, static_x)
        final_logits = base_logits + residual_logits
        bce_loss = criterion(final_logits, targets)
        base_probs = torch.sigmoid(base_logits)
        final_probs = torch.sigmoid(final_logits)
        residual_targets = targets - base_probs
        residual_preds = final_probs - base_probs
        mse_loss = torch.mean((residual_preds - residual_targets).pow(2))
        loss = bce_loss + config.residual_mse_weight * mse_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_bce += bce_loss.item() * batch_size
        total_mse += mse_loss.item() * batch_size
        predictions = (torch.sigmoid(final_logits) >= config.decision_threshold).to(targets.dtype)
        total_correct += int((predictions == targets).sum().item())
        total_count += batch_size

    return {
        "loss": total_loss / max(total_count, 1),
        "bce_loss": total_bce / max(total_count, 1),
        "mse_loss": total_mse / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
    }


@torch.no_grad()
def evaluate(model, dataloader, criterion, config: GRUGlobalClusterResidualConfig):
    model.eval()
    total_loss = total_bce = total_mse = 0.0
    total_correct = total_count = 0
    residual_all = []
    final_all = []
    target_all = []
    for global_seq, group_seq, static_x, base_logits, targets in dataloader:
        global_seq = global_seq.to(config.device)
        group_seq = group_seq.to(config.device)
        static_x = static_x.to(config.device)
        base_logits = base_logits.to(config.device)
        targets = targets.to(config.device)

        residual_logits = model(global_seq, group_seq, static_x)
        final_logits = base_logits + residual_logits
        bce_loss = criterion(final_logits, targets)
        base_probs = torch.sigmoid(base_logits)
        final_probs = torch.sigmoid(final_logits)
        residual_targets = targets - base_probs
        residual_preds = final_probs - base_probs
        mse_loss = torch.mean((residual_preds - residual_targets).pow(2))
        loss = bce_loss + config.residual_mse_weight * mse_loss

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_bce += bce_loss.item() * batch_size
        total_mse += mse_loss.item() * batch_size
        predictions = (final_probs >= config.decision_threshold).to(targets.dtype)
        total_correct += int((predictions == targets).sum().item())
        total_count += batch_size

        residual_all.append(residual_logits.detach().cpu().numpy())
        final_all.append(final_logits.detach().cpu().numpy())
        target_all.append(targets.detach().cpu().numpy())

    return {
        "loss": total_loss / max(total_count, 1),
        "bce_loss": total_bce / max(total_count, 1),
        "mse_loss": total_mse / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
        "residual_logits": np.concatenate(residual_all) if residual_all else np.array([], dtype=np.float32),
        "final_logits": np.concatenate(final_all) if final_all else np.array([], dtype=np.float32),
        "targets": np.concatenate(target_all) if target_all else np.array([], dtype=np.float32),
    }


@torch.no_grad()
def predict_residual_scores(
    model: ClusterResidualGRU,
    global_sequences: np.ndarray,
    group_sequences: np.ndarray,
    static_features: np.ndarray,
    config: GRUGlobalClusterResidualConfig,
) -> np.ndarray:
    model.eval()
    tensor_global = torch.tensor(global_sequences, dtype=torch.float32, device=config.device)
    tensor_group = torch.tensor(group_sequences, dtype=torch.float32, device=config.device)
    tensor_static = torch.tensor(static_features, dtype=torch.float32, device=config.device)
    return model(tensor_global, tensor_group, tensor_static).detach().cpu().numpy()


def fit_cluster_residual_fold(
    train_global: np.ndarray,
    train_group: np.ndarray,
    train_static: np.ndarray,
    train_base: np.ndarray,
    train_targets: np.ndarray,
    valid_global: np.ndarray,
    valid_group: np.ndarray,
    valid_static: np.ndarray,
    valid_base: np.ndarray,
    valid_targets: np.ndarray,
    config: GRUGlobalClusterResidualConfig,
    seed: int,
):
    seed_everything(seed)
    model = ClusterResidualGRU(config).to(config.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_loader = build_dataloader(
        train_global,
        train_group,
        train_static,
        train_base,
        train_targets,
        config,
        shuffle=True,
    )
    valid_loader = build_dataloader(
        valid_global,
        valid_group,
        valid_static,
        valid_base,
        valid_targets,
        config,
        shuffle=False,
    )

    best_state = None
    best_metrics = None
    best_score = -np.inf
    stale_epochs = 0

    for _epoch in range(config.max_epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, config)
        valid_metrics = evaluate(model, valid_loader, criterion, config)
        valid_score = valid_metrics["accuracy"]
        if valid_score > best_score:
            best_score = valid_score
            best_metrics = valid_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is None or best_metrics is None:
        raise RuntimeError("Cluster residual GRU training produced no checkpoint.")

    return best_state, best_metrics


def train_gru_global_cluster_residual(
    x_train_path: str | Path,
    y_train_path: str | Path,
    x_test_path: str | Path,
    output_dir: str | Path,
    preds_output_path: str | Path | None = None,
    n_clusters: int = N_CLUSTERS,
    global_config: GRUGlobalConfig | None = None,
    residual_config: GRUGlobalClusterResidualConfig | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_output_dir = output_dir / "global_base"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(SEED_VALUES[0])
    residual_config = residual_config or GRUGlobalClusterResidualConfig()
    residual_config.device = _resolve_device(residual_config.device)

    x_train_raw = pd.read_csv(x_train_path, index_col="ROW_ID")
    y_train = pd.read_csv(y_train_path, index_col="ROW_ID")
    x_test_raw = pd.read_csv(x_test_path, index_col="ROW_ID")

    global_config = global_config or GRUGlobalConfig(
        allocation_values=tuple(sorted(x_train_raw["ALLOCATION"].astype(str).unique().tolist())),
        device=residual_config.device,
        use_tqdm=True,
    )
    global_config.device = _resolve_device(global_config.device)

    x_train_features = build_gru_timeseries_features(x_train_raw)
    fold_metrics_base, oof_base = fit_and_save_global_gru(
        x_df=x_train_features,
        y_df=y_train,
        config=global_config,
        output_dir=base_output_dir,
        seed_values=SEED_VALUES,
    )
    fold_metrics_base.to_csv(base_output_dir / "fold_metrics_all_groups.csv", index=False)
    oof_base.to_csv(base_output_dir / "oof_predictions.csv", index=False)

    base_test = predict_global_gru(
        x_test_path=x_test_path,
        x_train_path=x_train_path,
        model_dir=base_output_dir,
        seed_values=SEED_VALUES,
        device=residual_config.device,
    ).set_index("ROW_ID")

    allocation_to_cluster, cluster_frame = build_allocation_cluster_map(
        x_train_path=x_train_path,
        n_clusters=n_clusters,
        random_state=CLUSTER_RANDOM_STATE,
    )

    x_train = build_gru_timeseries_features(x_train_raw)
    x_test = build_gru_timeseries_features(x_test_raw)
    x_train["ALLOCATION"] = x_train["ALLOCATION"].astype(str)
    x_test["ALLOCATION"] = x_test["ALLOCATION"].astype(str)
    x_train["CLUSTER"] = x_train["ALLOCATION"].map(allocation_to_cluster).astype(int).astype(str)
    x_test["CLUSTER"] = x_test["ALLOCATION"].map(allocation_to_cluster).astype(int).astype(str)

    global_cols, group_cols, static_cols = get_gru_global_feature_columns()
    feature_cols = list(dict.fromkeys([*global_cols, *group_cols, *static_cols]))
    prepared_train, scaler = prepare_dataframe(x_train, feature_cols, residual_config, scaler=None, fit_scaler=True)
    prepared_test, _ = prepare_dataframe(x_test, feature_cols, residual_config, scaler=scaler, fit_scaler=False)
    train_global, train_group, train_static = dataframe_to_inputs(prepared_train)
    test_global, test_group, test_static = dataframe_to_inputs(prepared_test)

    train_target_binary = (y_train["target"].to_numpy(dtype=np.float32) >= 0.0).astype(np.float32)
    base_oof_logits = oof_base.set_index("ROW_ID").loc[prepared_train.index, "final_logit_mean"].to_numpy(dtype=np.float32)
    base_test_logits = base_test.loc[prepared_test.index, "score_mean"].to_numpy(dtype=np.float32)

    oof_residual = pd.DataFrame(index=prepared_train.index)
    test_residual = pd.DataFrame(index=prepared_test.index)
    fold_metrics = []
    manifest_rows = []
    cluster_values = sorted(prepared_train["CLUSTER"].unique().tolist())

    for seed in SEED_VALUES:
        seed_resid_oof = pd.Series(index=prepared_train.index, dtype=np.float64)
        seed_resid_test = pd.Series(index=prepared_test.index, dtype=np.float64)

        for cluster_value in tqdm(cluster_values, desc=f"[cluster residual] seed={seed} clusters", leave=True):
            train_mask = prepared_train["CLUSTER"] == cluster_value
            test_mask = prepared_test["CLUSTER"] == cluster_value
            cluster_train = prepared_train.loc[train_mask].copy()
            cluster_test = prepared_test.loc[test_mask].copy()
            cluster_train_idx = cluster_train.index.to_numpy()
            cluster_test_idx = cluster_test.index.to_numpy()
            train_pos = prepared_train.index.get_indexer(cluster_train_idx)
            test_pos = prepared_test.index.get_indexer(cluster_test_idx)

            fold_assignments = build_time_folds(cluster_train, n_splits=residual_config.n_splits, ts_col=residual_config.ts_col)
            test_fold_scores = []

            for fold_id in tqdm(range(residual_config.n_splits), desc=f"  cluster={cluster_value} folds", leave=False):
                local_train_mask = (fold_assignments != fold_id).to_numpy()
                local_valid_mask = (fold_assignments == fold_id).to_numpy()

                state_dict, metrics = fit_cluster_residual_fold(
                    train_global=train_global[train_pos][local_train_mask],
                    train_group=train_group[train_pos][local_train_mask],
                    train_static=train_static[train_pos][local_train_mask],
                    train_base=base_oof_logits[train_pos][local_train_mask],
                    train_targets=train_target_binary[train_pos][local_train_mask],
                    valid_global=train_global[train_pos][local_valid_mask],
                    valid_group=train_group[train_pos][local_valid_mask],
                    valid_static=train_static[train_pos][local_valid_mask],
                    valid_base=base_oof_logits[train_pos][local_valid_mask],
                    valid_targets=train_target_binary[train_pos][local_valid_mask],
                    config=residual_config,
                    seed=seed + fold_id,
                )

                model = ClusterResidualGRU(residual_config).to(residual_config.device)
                model.load_state_dict(state_dict)
                valid_scores = predict_residual_scores(
                    model,
                    train_global[train_pos][local_valid_mask],
                    train_group[train_pos][local_valid_mask],
                    train_static[train_pos][local_valid_mask],
                    residual_config,
                )
                seed_resid_oof.loc[cluster_train.index[local_valid_mask]] = valid_scores

                if len(cluster_test_idx) > 0:
                    test_fold_scores.append(
                        predict_residual_scores(
                            model,
                            test_global[test_pos],
                            test_group[test_pos],
                            test_static[test_pos],
                            residual_config,
                        )
                    )

                final_valid_logits = base_oof_logits[train_pos][local_valid_mask] + valid_scores
                final_valid_pred = (1.0 / (1.0 + np.exp(-final_valid_logits)) >= residual_config.decision_threshold).astype(int)
                fold_metrics.append(
                    {
                        "seed": seed,
                        "cluster": cluster_value,
                        "fold": fold_id,
                        "n_train": int(local_train_mask.sum()),
                        "n_valid": int(local_valid_mask.sum()),
                        "valid_accuracy": float((final_valid_pred == train_target_binary[train_pos][local_valid_mask]).mean()),
                        "valid_loss": float(metrics["loss"]),
                        "valid_bce_loss": float(metrics["bce_loss"]),
                        "valid_mse_loss": float(metrics["mse_loss"]),
                    }
                )

                model_path = output_dir / f"cluster_residual_{cluster_value}_seed{seed}_fold{fold_id}.pt"
                torch.save(
                    {
                        "config": asdict(residual_config),
                        "state_dict": state_dict,
                        "cluster": cluster_value,
                        "seed": seed,
                        "fold": fold_id,
                    },
                    model_path,
                )
                manifest_rows.append(
                    {
                        "stage": "cluster_residual",
                        "cluster": cluster_value,
                        "seed": seed,
                        "fold": fold_id,
                        "model_path": model_path.name,
                    }
                )

            if len(cluster_test_idx) > 0 and test_fold_scores:
                seed_resid_test.loc[cluster_test.index] = np.mean(np.vstack(test_fold_scores), axis=0)

        oof_residual[f"residual_logit_seed_{seed}"] = seed_resid_oof.reindex(oof_residual.index)
        test_residual[f"residual_logit_seed_{seed}"] = seed_resid_test.reindex(test_residual.index)

    if oof_residual.isna().any().any():
        missing_cols = oof_residual.columns[oof_residual.isna().any()].tolist()
        raise RuntimeError(f"Missing residual OOF predictions in columns: {missing_cols}")

    oof_output = pd.DataFrame(index=prepared_train.index)
    oof_output["base_logit"] = base_oof_logits
    oof_output["residual_logit"] = oof_residual.mean(axis=1)
    oof_output["final_logit"] = oof_output["base_logit"] + oof_output["residual_logit"]
    oof_output["probability"] = 1.0 / (1.0 + np.exp(-oof_output["final_logit"]))
    oof_output["prediction"] = (oof_output["probability"] >= residual_config.decision_threshold).astype(int)
    oof_output["target"] = y_train.loc[oof_output.index, "target"].to_numpy()
    oof_output["target_binary"] = (oof_output["target"].to_numpy() >= 0.0).astype(int)
    oof_output["GROUP"] = x_train_raw.loc[oof_output.index, "GROUP"].to_numpy()
    oof_output["ALLOCATION"] = x_train_raw.loc[oof_output.index, "ALLOCATION"].astype(str).to_numpy()
    oof_output["CLUSTER"] = prepared_train.loc[oof_output.index, "CLUSTER"].to_numpy()
    oof_output["ROW_ID"] = oof_output.index

    test_output = pd.DataFrame(index=prepared_test.index)
    test_output["base_logit"] = base_test_logits
    test_output["residual_logit"] = test_residual.mean(axis=1)
    test_output["final_logit"] = test_output["base_logit"] + test_output["residual_logit"]
    test_output["probability"] = 1.0 / (1.0 + np.exp(-test_output["final_logit"]))
    test_output["prediction"] = (test_output["probability"] >= residual_config.decision_threshold).astype(int)
    test_output["GROUP"] = x_test_raw.loc[test_output.index, "GROUP"].to_numpy()
    test_output["ALLOCATION"] = x_test_raw.loc[test_output.index, "ALLOCATION"].astype(str).to_numpy()
    test_output["CLUSTER"] = prepared_test.loc[test_output.index, "CLUSTER"].to_numpy()
    test_output["ROW_ID"] = test_output.index

    base_oof_accuracy = float(((1.0 / (1.0 + np.exp(-base_oof_logits)) >= residual_config.decision_threshold).astype(int) == oof_output["target_binary"].to_numpy()).mean())
    final_oof_accuracy = float((oof_output["prediction"].to_numpy() == oof_output["target_binary"].to_numpy()).mean())

    comparison_df = pd.DataFrame(
        [
            {"model": "gru_global_base", "oof_accuracy": base_oof_accuracy},
            {"model": "gru_global_cluster_residual", "oof_accuracy": final_oof_accuracy},
        ]
    )
    fold_metrics_df = pd.DataFrame(fold_metrics)
    manifest = {
        "n_clusters": n_clusters,
        "cluster_random_state": CLUSTER_RANDOM_STATE,
        "seed_values": list(SEED_VALUES),
        "base_output_dir": str(base_output_dir),
        "residual_config": asdict(residual_config),
        "global_config": asdict(global_config),
        "models": manifest_rows,
    }

    oof_output.reset_index(drop=True).to_csv(output_dir / "oof_predictions.csv", index=False)
    test_output.reset_index(drop=True).to_csv(output_dir / "test_predictions_with_scores.csv", index=False)
    fold_metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    comparison_df.to_csv(output_dir / "comparison.csv", index=False)
    cluster_frame.to_csv(output_dir / "cluster_frame.csv", index=False)
    with open(output_dir / "gru_global_cluster_residual_manifest.json", "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)

    if preds_output_path is None:
        preds_output_path = output_dir / "preds_gru_global_cluster_residual.csv"
    preds_output_path = Path(preds_output_path)
    submission = test_output.loc[:, ["ROW_ID", "prediction"]].reset_index(drop=True)
    submission.to_csv(preds_output_path, index=False)

    return {
        "oof_output": oof_output.reset_index(drop=True),
        "test_output": test_output.reset_index(drop=True),
        "comparison_df": comparison_df,
        "fold_metrics_df": fold_metrics_df,
        "submission_df": submission,
        "cluster_frame": cluster_frame,
        "base_output_dir": base_output_dir,
        "output_dir": output_dir,
        "preds_output_path": preds_output_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train global GRU + cluster residual GRU.")
    parser.add_argument("--x-train", default="data/X_train.csv")
    parser.add_argument("--y-train", default="data/y_train.csv")
    parser.add_argument("--x-test", default="data/X_test.csv")
    parser.add_argument("--output-dir", default="model_gru_global_cluster_residual")
    parser.add_argument("--preds-output", default="code/preds_gru_global_cluster_residual.csv")
    parser.add_argument("--n-clusters", type=int, default=N_CLUSTERS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GRUGlobalClusterResidualConfig(device=args.device)
    result = train_gru_global_cluster_residual(
        x_train_path=args.x_train,
        y_train_path=args.y_train,
        x_test_path=args.x_test,
        output_dir=args.output_dir,
        preds_output_path=args.preds_output,
        n_clusters=args.n_clusters,
        residual_config=config,
    )
    print(result["comparison_df"].to_string(index=False))


if __name__ == "__main__":
    main()
