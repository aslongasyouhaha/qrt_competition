from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from utils import apply_standard_scaler, build_time_balanced_folds, fit_standard_scaler, seed_everything

GLOBAL_SEQUENCE_FEATURE_TEMPLATES = (
    "RET_{lag}",
    "SIGNED_VOLUME_{lag}",
    "RET_X_SIGNED_VOLUME_{lag}",
    "RET_X_ABS_SIGNED_VOLUME_{lag}",
    "SIGNED_VOLUME_OVER_TURNOVER_{lag}",
    "RET_X_ABS_SIGNED_VOLUME_OVER_TURNOVER_{lag}",
    "TS_MEAN_RET_{lag}",
    "TS_STD_RET_{lag}",
    "TS_MEAN_SIGNED_VOLUME_{lag}",
    "TS_STD_SIGNED_VOLUME_{lag}",
    "Z_TS_RET_{lag}",
    "Z_TS_SIGNED_VOLUME_{lag}",
    "TS_GROUP_MEAN_RET_{lag}",
    "TS_GROUP_MEAN_SIGNED_VOLUME_{lag}",
)

GROUP_SEQUENCE_FEATURE_TEMPLATES = (
    "RET_{lag}",
    "SIGNED_VOLUME_{lag}",
    "RET_X_SIGNED_VOLUME_{lag}",
    "RET_X_ABS_SIGNED_VOLUME_{lag}",
    "TS_GROUP_MEAN_RET_{lag}",
    "TS_GROUP_STD_RET_{lag}",
    "TS_GROUP_MEAN_SIGNED_VOLUME_{lag}",
    "TS_GROUP_STD_SIGNED_VOLUME_{lag}",
    "Z_TS_GROUP_RET_{lag}",
    "Z_TS_GROUP_SIGNED_VOLUME_{lag}",
    "TS_GROUP_MEAN_RET_X_SIGNED_VOLUME_{lag}",
    "TS_GROUP_STD_RET_X_SIGNED_VOLUME_{lag}",
    "Z_TS_GROUP_RET_X_SIGNED_VOLUME_{lag}",
    "TS_GROUP_MEAN_RET_X_ABS_SIGNED_VOLUME_{lag}",
    "TS_GROUP_STD_RET_X_ABS_SIGNED_VOLUME_{lag}",
    "Z_TS_GROUP_RET_X_ABS_SIGNED_VOLUME_{lag}",
)

STATIC_FEATURE_COLS = ("MEDIAN_DAILY_TURNOVER",)
GROUP_TO_INDEX = {1: 0, 2: 1, 3: 2, 4: 3}


@dataclass
class GRUGlobalConfig:
    group_col: str = "GROUP"
    allocation_col: str = "ALLOCATION"
    ts_col: str = "TS"
    fold_col: str = "fold"
    target_col: str = "target"
    n_splits: int = 5
    sequence_length: int = 20
    global_input_size: int = len(GLOBAL_SEQUENCE_FEATURE_TEMPLATES)
    group_input_size: int = len(GROUP_SEQUENCE_FEATURE_TEMPLATES)
    static_input_size: int = len(STATIC_FEATURE_COLS)
    num_groups: int = len(GROUP_TO_INDEX)
    allocation_values: tuple[str, ...] = ()
    allocation_vocab_size: int = 1
    allocation_embedding_dim: int = 16
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 20
    patience: int = 4
    scheduler_eta_min: float = 1e-6
    grad_clip_norm: float = 1.0
    fillna_value: float = 0.0
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    scale_features: bool = True
    metric_name: str = "accuracy"
    use_tqdm: bool = True
    decision_threshold: float = 0.5
    base_loss_weight: float = 0.25
    residual_loss_weight: float = 1.0


def normalize_gru_config(config: GRUGlobalConfig) -> GRUGlobalConfig:
    config.global_input_size = len(GLOBAL_SEQUENCE_FEATURE_TEMPLATES)
    config.group_input_size = len(GROUP_SEQUENCE_FEATURE_TEMPLATES)
    config.static_input_size = len(STATIC_FEATURE_COLS)
    config.num_groups = len(GROUP_TO_INDEX)
    config.allocation_vocab_size = max(1, len(config.allocation_values) + 1)
    return config


def build_gru_timeseries_features(df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    result = df.copy() if copy else df
    eps = np.float32(1e-6)
    turnover = result["MEDIAN_DAILY_TURNOVER"].astype(np.float32)
    ts_keys = [result["TS"]]
    tg_keys = [result["TS"], result["GROUP"]]
    feature_data: dict[str, pd.Series] = {}

    for lag in range(1, 21):
        ret = result[f"RET_{lag}"].astype(np.float32)
        vol = result[f"SIGNED_VOLUME_{lag}"].astype(np.float32)
        abs_vol = np.abs(vol).astype(np.float32)
        safe_turnover = turnover + eps

        ret_ts = ret.groupby(ts_keys)
        vol_ts = vol.groupby(ts_keys)
        ret_tg = ret.groupby(tg_keys)
        vol_tg = vol.groupby(tg_keys)

        ts_mean_ret = ret_ts.transform("mean").astype(np.float32)
        ts_std_ret = ret_ts.transform("std").fillna(0.0).astype(np.float32)
        ts_mean_vol = vol_ts.transform("mean").astype(np.float32)
        ts_std_vol = vol_ts.transform("std").fillna(0.0).astype(np.float32)
        tg_mean_ret = ret_tg.transform("mean").astype(np.float32)
        tg_std_ret = ret_tg.transform("std").fillna(0.0).astype(np.float32)
        tg_mean_vol = vol_tg.transform("mean").astype(np.float32)
        tg_std_vol = vol_tg.transform("std").fillna(0.0).astype(np.float32)

        feature_data[f"TS_MEAN_RET_{lag}"] = ts_mean_ret
        feature_data[f"TS_STD_RET_{lag}"] = ts_std_ret
        feature_data[f"Z_TS_RET_{lag}"] = ((ret - ts_mean_ret) / (ts_std_ret + eps)).astype(np.float32)
        feature_data[f"TS_MEAN_SIGNED_VOLUME_{lag}"] = ts_mean_vol
        feature_data[f"TS_STD_SIGNED_VOLUME_{lag}"] = ts_std_vol
        feature_data[f"Z_TS_SIGNED_VOLUME_{lag}"] = ((vol - ts_mean_vol) / (ts_std_vol + eps)).astype(np.float32)

        feature_data[f"TS_GROUP_MEAN_RET_{lag}"] = tg_mean_ret
        feature_data[f"TS_GROUP_STD_RET_{lag}"] = tg_std_ret
        feature_data[f"Z_TS_GROUP_RET_{lag}"] = ((ret - tg_mean_ret) / (tg_std_ret + eps)).astype(np.float32)
        feature_data[f"TS_GROUP_MEAN_SIGNED_VOLUME_{lag}"] = tg_mean_vol
        feature_data[f"TS_GROUP_STD_SIGNED_VOLUME_{lag}"] = tg_std_vol
        feature_data[f"Z_TS_GROUP_SIGNED_VOLUME_{lag}"] = ((vol - tg_mean_vol) / (tg_std_vol + eps)).astype(np.float32)

        signed_interaction = (ret * vol).astype(np.float32)
        abs_interaction = (ret * abs_vol).astype(np.float32)
        feature_data[f"RET_X_SIGNED_VOLUME_{lag}"] = signed_interaction
        feature_data[f"RET_X_ABS_SIGNED_VOLUME_{lag}"] = abs_interaction

        signed_tg = signed_interaction.groupby(tg_keys)
        abs_tg = abs_interaction.groupby(tg_keys)
        signed_tg_mean = signed_tg.transform("mean").astype(np.float32)
        signed_tg_std = signed_tg.transform("std").fillna(0.0).astype(np.float32)
        abs_tg_mean = abs_tg.transform("mean").astype(np.float32)
        abs_tg_std = abs_tg.transform("std").fillna(0.0).astype(np.float32)

        feature_data[f"TS_GROUP_MEAN_RET_X_SIGNED_VOLUME_{lag}"] = signed_tg_mean
        feature_data[f"TS_GROUP_STD_RET_X_SIGNED_VOLUME_{lag}"] = signed_tg_std
        feature_data[f"Z_TS_GROUP_RET_X_SIGNED_VOLUME_{lag}"] = (
            (signed_interaction - signed_tg_mean) / (signed_tg_std + eps)
        ).astype(np.float32)
        feature_data[f"TS_GROUP_MEAN_RET_X_ABS_SIGNED_VOLUME_{lag}"] = abs_tg_mean
        feature_data[f"TS_GROUP_STD_RET_X_ABS_SIGNED_VOLUME_{lag}"] = abs_tg_std
        feature_data[f"Z_TS_GROUP_RET_X_ABS_SIGNED_VOLUME_{lag}"] = (
            (abs_interaction - abs_tg_mean) / (abs_tg_std + eps)
        ).astype(np.float32)

        feature_data[f"SIGNED_VOLUME_OVER_TURNOVER_{lag}"] = (vol / safe_turnover).astype(np.float32)
        feature_data[f"RET_X_ABS_SIGNED_VOLUME_OVER_TURNOVER_{lag}"] = (ret * abs_vol / safe_turnover).astype(
            np.float32
        )

    feature_frame = pd.DataFrame(feature_data, index=result.index)
    return pd.concat([result, feature_frame], axis=1)


def get_sequence_feature_columns(templates: tuple[str, ...]) -> list[list[str]]:
    return [[template.format(lag=lag) for template in templates] for lag in range(20, 0, -1)]


def get_gru_global_feature_columns() -> tuple[list[str], list[str], list[str]]:
    global_seq = [c for cols in get_sequence_feature_columns(GLOBAL_SEQUENCE_FEATURE_TEMPLATES) for c in cols]
    group_seq = [c for cols in get_sequence_feature_columns(GROUP_SEQUENCE_FEATURE_TEMPLATES) for c in cols]
    static_cols = list(STATIC_FEATURE_COLS)
    return global_seq, group_seq, static_cols


def prepare_gru_dataframe(df: pd.DataFrame, feature_cols: list[str], config: GRUGlobalConfig, scaler=None, fit_scaler: bool = False):
    prepared = df.copy()
    prepared.loc[:, feature_cols] = prepared.loc[:, feature_cols].fillna(config.fillna_value)
    fitted_scaler = scaler
    if config.scale_features:
        if fit_scaler:
            fitted_scaler = fit_standard_scaler(prepared, feature_cols)
        elif scaler is None:
            raise ValueError("A scaler must be provided when fit_scaler is False.")
        prepared = apply_standard_scaler(prepared, fitted_scaler, feature_cols, copy=False)
    return prepared, fitted_scaler


def allocation_to_index(series: pd.Series, config: GRUGlobalConfig) -> np.ndarray:
    mapping = {value: idx + 1 for idx, value in enumerate(config.allocation_values)}
    return series.astype(str).map(mapping).fillna(0).to_numpy(dtype=np.int64)


def dataframe_to_gru_inputs(df: pd.DataFrame, config: GRUGlobalConfig):
    global_grouped = get_sequence_feature_columns(GLOBAL_SEQUENCE_FEATURE_TEMPLATES)
    group_grouped = get_sequence_feature_columns(GROUP_SEQUENCE_FEATURE_TEMPLATES)
    global_sequences = np.stack([df.loc[:, cols].to_numpy(dtype=np.float32) for cols in global_grouped], axis=1)
    group_sequences = np.stack([df.loc[:, cols].to_numpy(dtype=np.float32) for cols in group_grouped], axis=1)
    static_features = df.loc[:, list(STATIC_FEATURE_COLS)].to_numpy(dtype=np.float32)
    group_indices = df["GROUP"].astype(int).map(GROUP_TO_INDEX).to_numpy(dtype=np.int64)
    allocation_indices = allocation_to_index(df["ALLOCATION"], config)
    return global_sequences, group_sequences, static_features, group_indices, allocation_indices


class GRUResidualDataset(Dataset):
    def __init__(self, global_sequences, group_sequences, static_features, group_indices, allocation_indices, targets) -> None:
        self.global_sequences = torch.tensor(global_sequences, dtype=torch.float32)
        self.group_sequences = torch.tensor(group_sequences, dtype=torch.float32)
        self.static_features = torch.tensor(static_features, dtype=torch.float32)
        self.group_indices = torch.tensor(group_indices, dtype=torch.long)
        self.allocation_indices = torch.tensor(allocation_indices, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            self.global_sequences[idx],
            self.group_sequences[idx],
            self.static_features[idx],
            self.group_indices[idx],
            self.allocation_indices[idx],
            self.targets[idx],
        )


def build_residual_dataloader(global_sequences, group_sequences, static_features, group_indices, allocation_indices, targets, config: GRUGlobalConfig, shuffle: bool):
    dataset = GRUResidualDataset(global_sequences, group_sequences, static_features, group_indices, allocation_indices, targets)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
    )


class ResidualGroupGRU(nn.Module):
    def __init__(self, config: GRUGlobalConfig) -> None:
        super().__init__()
        config = normalize_gru_config(config)
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
        self.group_embedding = nn.Embedding(config.num_groups, config.hidden_size)
        self.global_allocation_embedding = nn.Embedding(config.allocation_vocab_size, config.hidden_size)
        self.group_allocation_embedding = nn.Embedding(config.allocation_vocab_size, config.hidden_size)
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
        self.base_head = nn.Linear(config.hidden_size, 1)
        self.group_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(config.hidden_size),
                    nn.Linear(config.hidden_size, config.hidden_size // 2),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_size // 2, 1),
                )
                for _ in range(config.num_groups)
            ]
        )

    def forward(self, global_sequences, group_sequences, static_features, group_indices, allocation_indices):
        global_output, _ = self.global_gru(global_sequences)
        group_output, _ = self.group_gru(group_sequences)
        global_repr = self.global_fuse(
            self.dropout(global_output[:, -1, :])
            + self.global_static_proj(static_features)
            + self.group_embedding(group_indices)
            + self.global_allocation_embedding(allocation_indices)
        )
        group_repr = self.group_fuse(
            self.dropout(group_output[:, -1, :])
            + self.group_static_proj(static_features)
            + self.group_allocation_embedding(allocation_indices)
        )
        base_scores = self.base_head(global_repr).squeeze(-1)
        residual_scores = torch.cat([head(group_repr) for head in self.group_heads], dim=1)
        selected_residual = residual_scores.gather(1, group_indices.view(-1, 1)).squeeze(1)
        final_scores = base_scores + selected_residual
        return final_scores, base_scores, selected_residual


def train_one_epoch(model, dataloader, criterion, optimizer, config: GRUGlobalConfig):
    model.train()
    total_loss = total_base_loss = total_residual_loss = 0.0
    total_correct = total_count = 0
    for global_seq, group_seq, static_x, group_idx, alloc_idx, targets in dataloader:
        global_seq = global_seq.to(config.device)
        group_seq = group_seq.to(config.device)
        static_x = static_x.to(config.device)
        group_idx = group_idx.to(config.device)
        alloc_idx = alloc_idx.to(config.device)
        targets = targets.to(config.device)
        optimizer.zero_grad(set_to_none=True)
        final_logits, base_logits, _ = model(global_seq, group_seq, static_x, group_idx, alloc_idx)
        final_loss = criterion(final_logits, targets)
        base_loss = criterion(base_logits, targets)
        base_probs = torch.sigmoid(base_logits.detach())
        final_probs = torch.sigmoid(final_logits)
        residual_targets = targets - base_probs
        residual_probs = final_probs - base_probs
        residual_loss = torch.mean((residual_probs - residual_targets).pow(2))
        loss = final_loss + config.base_loss_weight * base_loss + config.residual_loss_weight * residual_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_base_loss += base_loss.item() * batch_size
        total_residual_loss += residual_loss.item() * batch_size
        final_predictions = (torch.sigmoid(final_logits) >= config.decision_threshold).to(targets.dtype)
        total_correct += int((final_predictions == targets).sum().item())
        total_count += batch_size

    return {
        "loss": total_loss / max(total_count, 1),
        "base_loss": total_base_loss / max(total_count, 1),
        "residual_loss": total_residual_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
    }


@torch.no_grad()
def evaluate(model, dataloader, criterion, config: GRUGlobalConfig):
    model.eval()
    total_loss = total_base_loss = total_residual_loss = 0.0
    total_correct = total_base_correct = total_count = 0
    for global_seq, group_seq, static_x, group_idx, alloc_idx, targets in dataloader:
        global_seq = global_seq.to(config.device)
        group_seq = group_seq.to(config.device)
        static_x = static_x.to(config.device)
        group_idx = group_idx.to(config.device)
        alloc_idx = alloc_idx.to(config.device)
        targets = targets.to(config.device)
        final_logits, base_logits, _ = model(global_seq, group_seq, static_x, group_idx, alloc_idx)
        final_loss = criterion(final_logits, targets)
        base_loss = criterion(base_logits, targets)
        base_probs = torch.sigmoid(base_logits.detach())
        final_probs = torch.sigmoid(final_logits)
        residual_targets = targets - base_probs
        residual_probs = final_probs - base_probs
        residual_loss = torch.mean((residual_probs - residual_targets).pow(2))

        batch_size = targets.size(0)
        total_loss += final_loss.item() * batch_size
        total_base_loss += base_loss.item() * batch_size
        total_residual_loss += residual_loss.item() * batch_size
        final_predictions = (final_probs >= config.decision_threshold).to(targets.dtype)
        base_predictions = (torch.sigmoid(base_logits) >= config.decision_threshold).to(targets.dtype)
        total_correct += int((final_predictions == targets).sum().item())
        total_base_correct += int((base_predictions == targets).sum().item())
        total_count += batch_size

    return {
        "loss": total_loss / max(total_count, 1),
        "base_loss": total_base_loss / max(total_count, 1),
        "residual_loss": total_residual_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
        "base_accuracy": total_base_correct / max(total_count, 1),
    }


@torch.no_grad()
def predict_scores(model, global_sequences, group_sequences, static_features, group_indices, allocation_indices, config: GRUGlobalConfig):
    model.eval()
    tensor_global = torch.tensor(global_sequences, dtype=torch.float32, device=config.device)
    tensor_group = torch.tensor(group_sequences, dtype=torch.float32, device=config.device)
    tensor_static = torch.tensor(static_features, dtype=torch.float32, device=config.device)
    tensor_group_idx = torch.tensor(group_indices, dtype=torch.long, device=config.device)
    tensor_alloc_idx = torch.tensor(allocation_indices, dtype=torch.long, device=config.device)
    final_logits, base_logits, _ = model(tensor_global, tensor_group, tensor_static, tensor_group_idx, tensor_alloc_idx)
    final_probs = torch.sigmoid(final_logits)
    base_probs = torch.sigmoid(base_logits)
    return (
        final_logits.detach().cpu().numpy(),
        base_logits.detach().cpu().numpy(),
        final_probs.detach().cpu().numpy(),
        base_probs.detach().cpu().numpy(),
    )


def fit_single_fold(train_df, valid_df, y_train, y_valid, config: GRUGlobalConfig, scaler=None):
    config = normalize_gru_config(config)
    global_cols, group_cols, static_cols = get_gru_global_feature_columns()
    feature_cols = list(dict.fromkeys([*global_cols, *group_cols, *static_cols]))
    if scaler is None:
        train_prepared, scaler = prepare_gru_dataframe(train_df, feature_cols, config, fit_scaler=True)
    else:
        train_prepared, _ = prepare_gru_dataframe(train_df, feature_cols, config, scaler=scaler, fit_scaler=False)
    valid_prepared, _ = prepare_gru_dataframe(valid_df, feature_cols, config, scaler=scaler, fit_scaler=False)

    train_global, train_group, train_static, train_group_idx, train_alloc_idx = dataframe_to_gru_inputs(train_prepared, config)
    valid_global, valid_group, valid_static, valid_group_idx, valid_alloc_idx = dataframe_to_gru_inputs(valid_prepared, config)
    y_train_values = (y_train.to_numpy(dtype=np.float32) >= 0.0).astype(np.float32)
    y_valid_values = (y_valid.to_numpy(dtype=np.float32) >= 0.0).astype(np.float32)

    train_loader = build_residual_dataloader(train_global, train_group, train_static, train_group_idx, train_alloc_idx, y_train_values, config, shuffle=True)
    valid_loader = build_residual_dataloader(valid_global, valid_group, valid_static, valid_group_idx, valid_alloc_idx, y_valid_values, config, shuffle=False)

    model = ResidualGroupGRU(config).to(config.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs, eta_min=config.scheduler_eta_min)

    best_metric = -np.inf
    best_state = None
    best_metrics = {}
    patience_left = config.patience
    epoch_iterator = range(config.max_epochs)
    if config.use_tqdm:
        epoch_iterator = tqdm(epoch_iterator, desc="epochs", leave=False)

    for epoch in epoch_iterator:
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, config)
        valid_metrics = evaluate(model, valid_loader, criterion, config)
        metric_value = valid_metrics[config.metric_name]
        if config.use_tqdm:
            epoch_iterator.set_postfix(
                train_loss=f"{train_metrics['loss']:.4f}",
                valid_loss=f"{valid_metrics['loss']:.4f}",
                residual_loss=f"{valid_metrics['residual_loss']:.4f}",
                valid_acc=f"{valid_metrics['accuracy']:.4f}",
                base_acc=f"{valid_metrics['base_accuracy']:.4f}",
            )
        if metric_value > best_metric:
            best_metric = metric_value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "epoch": float(epoch + 1),
                "train_loss": train_metrics["loss"],
                "train_residual_loss": train_metrics["residual_loss"],
                "train_accuracy": train_metrics["accuracy"],
                "valid_loss": valid_metrics["loss"],
                "valid_residual_loss": valid_metrics["residual_loss"],
                "valid_accuracy": valid_metrics["accuracy"],
                "valid_base_accuracy": valid_metrics["base_accuracy"],
            }
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
        scheduler.step()

    if best_state is None:
        raise RuntimeError("Training finished without a best checkpoint.")
    model.load_state_dict(best_state)
    return model, best_metrics


def fit_seed_cv(x_df, y_df, config: GRUGlobalConfig, seed: int, scaler):
    config = normalize_gru_config(config)
    fold_frame = build_time_balanced_folds(
        x_df[[config.ts_col, config.group_col, config.allocation_col]],
        n_splits=config.n_splits,
        seed=seed,
        ts_col=config.ts_col,
        group_col=config.group_col,
        allocation_col=config.allocation_col,
        fold_col=config.fold_col,
    )
    x_seed = x_df.copy()
    x_seed[config.fold_col] = fold_frame[config.fold_col].reindex(x_df.index)

    global_cols, group_cols, static_cols = get_gru_global_feature_columns()
    feature_cols = list(dict.fromkeys([*global_cols, *group_cols, *static_cols]))
    state_dicts = []
    fold_metrics = []
    oof_frame = pd.DataFrame(index=x_seed.index)

    for fold_id in range(config.n_splits):
        valid_mask = x_seed[config.fold_col] == fold_id
        train_mask = ~valid_mask
        x_train = x_seed.loc[train_mask].copy()
        x_valid = x_seed.loc[valid_mask].copy()
        y_train = y_df.loc[x_train.index, config.target_col].copy()
        y_valid = y_df.loc[x_valid.index, config.target_col].copy()
        model, metrics = fit_single_fold(x_train, x_valid, y_train, y_valid, config=config, scaler=scaler)
        state_dicts.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        valid_prepared, _ = prepare_gru_dataframe(x_valid, feature_cols, config, scaler=scaler, fit_scaler=False)
        valid_global, valid_group, valid_static, valid_group_idx, valid_alloc_idx = dataframe_to_gru_inputs(valid_prepared, config)
        final_logits, base_logits, final_probs, base_probs = predict_scores(model, valid_global, valid_group, valid_static, valid_group_idx, valid_alloc_idx, config)
        oof_frame.loc[x_valid.index, f"seed_{seed}_final_logit"] = final_logits
        oof_frame.loc[x_valid.index, f"seed_{seed}_base_logit"] = base_logits
        oof_frame.loc[x_valid.index, f"seed_{seed}_final_prob"] = final_probs
        oof_frame.loc[x_valid.index, f"seed_{seed}_base_prob"] = base_probs
        fold_metrics.append({"seed": seed, "fold": fold_id, **metrics})
        print(f"seed={seed} fold={fold_id} valid_accuracy={metrics['valid_accuracy']:.6f} valid_base_accuracy={metrics['valid_base_accuracy']:.6f}")

    return state_dicts, pd.DataFrame(fold_metrics), oof_frame


def save_seed_checkpoint(output_dir, seed, state_dicts, config: GRUGlobalConfig) -> Path:
    config = normalize_gru_config(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"gru_seed{seed}.pt"
    torch.save({"config": asdict(config), "state_dicts": state_dicts}, checkpoint_path)
    return checkpoint_path


def build_global_scaler(x_df: pd.DataFrame, feature_cols: list[str], fillna_value: float):
    prepared = x_df.copy()
    prepared.loc[:, feature_cols] = prepared.loc[:, feature_cols].fillna(fillna_value)
    return fit_standard_scaler(prepared, feature_cols)


def fit_and_save_global_gru(x_df, y_df, config: GRUGlobalConfig, output_dir, seed_values):
    config = normalize_gru_config(config)
    global_cols, group_cols, static_cols = get_gru_global_feature_columns()
    feature_cols = list(dict.fromkeys([*global_cols, *group_cols, *static_cols]))
    global_scaler = build_global_scaler(x_df=x_df, feature_cols=feature_cols, fillna_value=config.fillna_value)

    all_fold_metrics = []
    all_oof_frames = []
    for seed in seed_values:
        state_dicts, fold_metrics, seed_oof = fit_seed_cv(x_df, y_df, config, seed, global_scaler)
        save_seed_checkpoint(output_dir, seed, state_dicts, config)
        all_fold_metrics.append(fold_metrics)
        all_oof_frames.append(seed_oof)

    fold_metrics_df = pd.concat(all_fold_metrics, ignore_index=True)
    combined_oof = pd.concat(all_oof_frames, axis=1)
    final_logit_cols = [col for col in combined_oof.columns if col.endswith("_final_logit")]
    base_logit_cols = [col for col in combined_oof.columns if col.endswith("_base_logit")]
    final_prob_cols = [col for col in combined_oof.columns if col.endswith("_final_prob")]
    base_prob_cols = [col for col in combined_oof.columns if col.endswith("_base_prob")]
    combined_oof["final_logit_mean"] = combined_oof[final_logit_cols].mean(axis=1)
    combined_oof["base_logit_mean"] = combined_oof[base_logit_cols].mean(axis=1)
    combined_oof["final_prob_mean"] = combined_oof[final_prob_cols].mean(axis=1)
    combined_oof["base_prob_mean"] = combined_oof[base_prob_cols].mean(axis=1)
    combined_oof["target"] = y_df.loc[combined_oof.index, config.target_col].to_numpy()
    combined_oof["target_binary"] = (combined_oof["target"].to_numpy() >= 0.0).astype(int)
    combined_oof["prediction"] = (combined_oof["final_prob_mean"].to_numpy() >= config.decision_threshold).astype(int)
    combined_oof["base_prediction"] = (combined_oof["base_prob_mean"].to_numpy() >= config.decision_threshold).astype(int)
    combined_oof["ROW_ID"] = combined_oof.index
    return fold_metrics_df, combined_oof


def load_checkpoint(checkpoint_path, map_location="cpu"):
    return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def config_from_checkpoint(checkpoint: dict, device: str = "cpu") -> GRUGlobalConfig:
    raw_config = dict(checkpoint["config"])
    raw_config["device"] = device
    raw_config["use_tqdm"] = False
    return normalize_gru_config(GRUGlobalConfig(**raw_config))


@torch.no_grad()
def predict_seed_scores(x_df, checkpoint: dict, scaler, device="cpu"):
    config = config_from_checkpoint(checkpoint, device=device)
    global_cols, group_cols, static_cols = get_gru_global_feature_columns()
    feature_cols = list(dict.fromkeys([*global_cols, *group_cols, *static_cols]))
    prepared, _ = prepare_gru_dataframe(x_df, feature_cols, config, scaler=scaler, fit_scaler=False)
    global_seq, group_seq, static_x, group_idx, alloc_idx = dataframe_to_gru_inputs(prepared, config)
    fold_final_logits = []
    fold_base_logits = []
    fold_final_probs = []
    fold_base_probs = []
    for state_dict in checkpoint["state_dicts"]:
        model = ResidualGroupGRU(config).to(config.device)
        model.load_state_dict(state_dict)
        model.eval()
        final_logits, base_logits, final_probs, base_probs = predict_scores(model, global_seq, group_seq, static_x, group_idx, alloc_idx, config)
        fold_final_logits.append(final_logits)
        fold_base_logits.append(base_logits)
        fold_final_probs.append(final_probs)
        fold_base_probs.append(base_probs)
    return (
        np.mean(np.vstack(fold_final_logits), axis=0),
        np.mean(np.vstack(fold_base_logits), axis=0),
        np.mean(np.vstack(fold_final_probs), axis=0),
        np.mean(np.vstack(fold_base_probs), axis=0),
    )


def predict_global_gru(x_test_path, x_train_path, model_dir, seed_values, device="cpu"):
    model_dir = Path(model_dir)
    first_checkpoint = load_checkpoint(model_dir / f"gru_seed{seed_values[0]}.pt", map_location=device)
    base_config = config_from_checkpoint(first_checkpoint, device=device)

    x_train = pd.read_csv(x_train_path, index_col="ROW_ID")
    x_train = build_gru_timeseries_features(x_train)
    global_cols, group_cols, static_cols = get_gru_global_feature_columns()
    feature_cols = list(dict.fromkeys([*global_cols, *group_cols, *static_cols]))
    global_scaler = build_global_scaler(x_df=x_train, feature_cols=feature_cols, fillna_value=base_config.fillna_value)

    x_test = pd.read_csv(x_test_path, index_col="ROW_ID")
    x_test = build_gru_timeseries_features(x_test)
    predictions = pd.DataFrame(index=x_test.index)
    predictions["GROUP"] = x_test["GROUP"].astype(int)
    predictions["ALLOCATION"] = x_test["ALLOCATION"]
    for seed in seed_values:
        checkpoint = load_checkpoint(model_dir / f"gru_seed{seed}.pt", map_location=device)
        final_logits, base_logits, final_probs, base_probs = predict_seed_scores(x_test, checkpoint, global_scaler, device=device)
        predictions[f"final_logit_seed_{seed}"] = final_logits
        predictions[f"base_logit_seed_{seed}"] = base_logits
        predictions[f"final_prob_seed_{seed}"] = final_probs
        predictions[f"base_prob_seed_{seed}"] = base_probs
    final_logit_cols = [col for col in predictions.columns if col.startswith("final_logit_seed_")]
    base_logit_cols = [col for col in predictions.columns if col.startswith("base_logit_seed_")]
    final_prob_cols = [col for col in predictions.columns if col.startswith("final_prob_seed_")]
    base_prob_cols = [col for col in predictions.columns if col.startswith("base_prob_seed_")]
    predictions["score_mean"] = predictions[final_logit_cols].mean(axis=1)
    predictions["base_score_mean"] = predictions[base_logit_cols].mean(axis=1)
    predictions["probability_mean"] = predictions[final_prob_cols].mean(axis=1)
    predictions["base_probability_mean"] = predictions[base_prob_cols].mean(axis=1)
    predictions["prediction"] = (predictions["probability_mean"].to_numpy() >= base_config.decision_threshold).astype(int)
    predictions["ROW_ID"] = predictions.index
    return predictions.reset_index(drop=True)


def parse_seed_values(raw_seed_values: str) -> tuple[int, ...]:
    values = tuple(int(seed.strip()) for seed in raw_seed_values.split(",") if seed.strip())
    if not values:
        raise ValueError("At least one seed must be provided.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a global residual-head GRU with split feature branches.")
    parser.add_argument("--x-train", default="data/X_train.csv")
    parser.add_argument("--y-train", default="data/y_train.csv")
    parser.add_argument("--output-dir", default="model_gru_global")
    parser.add_argument("--seed-values", default="42,52,62")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_values = parse_seed_values(args.seed_values)
    seed_everything(seed_values[0])
    x_train = pd.read_csv(args.x_train, index_col="ROW_ID")
    y_train = pd.read_csv(args.y_train, index_col="ROW_ID")
    x_train = build_gru_timeseries_features(x_train)
    config = normalize_gru_config(
        GRUGlobalConfig(
            group_col="GROUP",
            allocation_col="ALLOCATION",
            ts_col="TS",
            fold_col="fold",
            target_col="target",
            n_splits=5,
            allocation_values=tuple(sorted(x_train["ALLOCATION"].astype(str).unique().tolist())),
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            batch_size=1024,
            learning_rate=1e-3,
            weight_decay=1e-4,
            max_epochs=20,
            patience=4,
            scheduler_eta_min=1e-6,
            grad_clip_norm=1.0,
            fillna_value=0.0,
            num_workers=0,
            device=args.device,
            scale_features=True,
            metric_name="accuracy",
            use_tqdm=True,
            base_loss_weight=0.25,
            residual_loss_weight=1.0,
        )
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics_df, oof_frame = fit_and_save_global_gru(x_train, y_train, config, output_dir, seed_values)
    fold_metrics_df.to_csv(output_dir / "fold_metrics_all_groups.csv", index=False)
    oof_frame.to_csv(output_dir / "oof_predictions.csv", index=False)
    final_oof_accuracy = float((oof_frame["prediction"].to_numpy() == oof_frame["target_binary"].to_numpy()).mean())
    base_oof_accuracy = float((oof_frame["base_prediction"].to_numpy() == oof_frame["target_binary"].to_numpy()).mean())
    print(f"saved residual-head GRU models to: {output_dir}")
    print("validation accuracy by seed:")
    print(fold_metrics_df.groupby("seed")["valid_accuracy"].mean().to_string())
    print(f"base_oof_accuracy: {base_oof_accuracy:.6f}")
    print(f"final_oof_accuracy: {final_oof_accuracy:.6f}")


if __name__ == "__main__":
    main()
