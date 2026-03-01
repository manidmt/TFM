'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-02-27

@description: Minimal FT-Transformer-like classifier for numeric tabular inputs.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import copy

import numpy as np

from .common import set_global_seed, to_numpy_x, to_numpy_y


@dataclass(frozen=True)
class FTTransformerConfig:
    d_token: int = 32
    n_heads: int = 4
    n_layers: int = 2
    ffn_multiplier: int = 2
    dropout: float = 0.1
    n_classes: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 30
    patience: int = 5
    seed: int = 42
    device: str = "cpu"


class _FTBackbone:
    def __init__(self, n_features: int, cfg: FTTransformerConfig):
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self, n_features_inner: int, cfg_inner: FTTransformerConfig):
                super().__init__()
                self.n_features = n_features_inner
                self.feature_embed = nn.Linear(1, cfg_inner.d_token)
                self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg_inner.d_token))
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=cfg_inner.d_token,
                    nhead=cfg_inner.n_heads,
                    dim_feedforward=cfg_inner.d_token * cfg_inner.ffn_multiplier,
                    dropout=cfg_inner.dropout,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=cfg_inner.n_layers,
                )
                self.head = nn.Sequential(
                    nn.LayerNorm(cfg_inner.d_token),
                    nn.Linear(cfg_inner.d_token, cfg_inner.d_token),
                    nn.GELU(),
                    nn.Dropout(cfg_inner.dropout),
                    nn.Linear(cfg_inner.d_token, cfg_inner.n_classes),
                )

            def forward(self, x):
                # x: [B, F] -> token embedding per feature [B, F, D]
                x_tokens = self.feature_embed(x.unsqueeze(-1))
                cls = self.cls_token.expand(x_tokens.size(0), -1, -1)
                tokens = torch.cat([cls, x_tokens], dim=1)
                tokens = self.encoder(tokens)
                cls_out = tokens[:, 0, :]
                return self.head(cls_out)

        self.model = Net(n_features, cfg)


class FTTransformerClassifier:
    def __init__(self, cfg: FTTransformerConfig):
        self.cfg = cfg
        self.net = None
        self.device = None
        self.x_mean = None
        self.x_std = None

    def _normalise(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) / self.x_std

    def _build(self, n_features: int):
        import torch

        backbone = _FTBackbone(n_features=n_features, cfg=self.cfg)
        self.net = backbone.model
        self.device = torch.device(self.cfg.device)
        self.net.to(self.device)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        set_global_seed(self.cfg.seed, use_torch=True)

        x_train = to_numpy_x(X_train, dtype=np.float32)
        y_train_np = to_numpy_y(y_train)
        x_valid = to_numpy_x(X_valid, dtype=np.float32) if X_valid is not None else None
        y_valid_np = to_numpy_y(y_valid) if y_valid is not None else None

        self.x_mean = x_train.mean(axis=0, keepdims=True)
        self.x_std = x_train.std(axis=0, keepdims=True)
        self.x_std = np.where(self.x_std < 1e-6, 1.0, self.x_std)

        x_train = self._normalise(x_train)
        if x_valid is not None:
            x_valid = self._normalise(x_valid)

        if self.net is None:
            self._build(n_features=x_train.shape[1])

        train_ds = TensorDataset(
            torch.from_numpy(x_train).float(),
            torch.from_numpy(y_train_np).long(),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )

        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        criterion = torch.nn.CrossEntropyLoss()

        best_state = copy.deepcopy(self.net.state_dict())
        best_score = float("inf")
        epochs_no_improve = 0

        for _ in range(self.cfg.max_epochs):
            self.net.train()
            train_loss = 0.0
            train_count = 0

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                logits = self.net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                bsz = xb.size(0)
                train_loss += float(loss.item()) * bsz
                train_count += int(bsz)

            score = train_loss / max(train_count, 1)
            if x_valid is not None and y_valid_np is not None:
                self.net.eval()
                with torch.no_grad():
                    xv = torch.from_numpy(x_valid).float().to(self.device)
                    yv = torch.from_numpy(y_valid_np).long().to(self.device)
                    v_logits = self.net(xv)
                    v_loss = criterion(v_logits, yv)
                    score = float(v_loss.item())

            if score + 1e-8 < best_score:
                best_score = score
                best_state = copy.deepcopy(self.net.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.cfg.patience:
                    break

        self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def predict_proba(self, X) -> np.ndarray:
        import torch

        if self.net is None:
            raise RuntimeError("Model is not fitted.")

        x = self._normalise(to_numpy_x(X, dtype=np.float32))
        with torch.no_grad():
            logits = self.net(torch.from_numpy(x).float().to(self.device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.asarray(np.argmax(probs, axis=1), dtype=int)


def make_model(cfg: FTTransformerConfig) -> FTTransformerClassifier:
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise ImportError("torch no está instalado. Instala con: poetry add torch") from e
    return FTTransformerClassifier(cfg=cfg)


def fit(
    model: FTTransformerClassifier,
    X_train,
    y_train,
    X_valid: Optional[np.ndarray] = None,
    y_valid: Optional[np.ndarray] = None,
) -> FTTransformerClassifier:
    return model.fit(X_train, y_train, X_valid=X_valid, y_valid=y_valid)


def predict(model: FTTransformerClassifier, X) -> np.ndarray:
    return model.predict(X)


def predict_proba(model: FTTransformerClassifier, X) -> np.ndarray:
    return model.predict_proba(X)


# TODO: Add categorical-token support and richer FT-Transformer blocks if needed.
