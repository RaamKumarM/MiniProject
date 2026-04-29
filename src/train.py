from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import Data

_ROOT = Path(__file__).resolve().parent.parent
_SRC = Path(__file__).resolve().parent
for p in (_ROOT, _SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import GAT_HEADS, GAT_HIDDEN_CHANNELS, GAT_OUT_CHANNELS
from model import DDoSGAT


def train_model(data: Data) -> dict[str, list[float]]:
    """Train DDoSGAT with balanced CE weights, oversampled attack train nodes, early stop on val F1."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    y_np = data.y.detach().cpu().numpy()
    n0 = int((y_np == 0).sum())
    n1 = int((y_np == 1).sum())
    print(
        f"Class distribution (all nodes): label=0: {n0}, label=1: {n1}, total: {len(y_np)}",
        flush=True,
    )

    classes = np.array([0, 1])
    uniq = np.unique(y_np)
    if len(uniq) < 2:
        class_weights = np.ones(2, dtype=np.float32)
        print("Single class in labels; using uniform class weights.", flush=True)
    else:
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_np,
        ).astype(np.float32)
    print(f"Balanced class weights: {class_weights}", flush=True)
    ce_weight = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=ce_weight)

    train_idx = torch.where(data.train_mask)[0]
    attack_train = train_idx[data.y[train_idx] == 1]
    oversample = attack_train.repeat(10)
    train_indices = torch.cat([train_idx, oversample], dim=0)

    model = DDoSGAT(
        in_channels=data.x.shape[1],
        hidden_channels=GAT_HIDDEN_CHANNELS,
        out_channels=GAT_OUT_CHANNELS,
        heads=GAT_HEADS,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=5e-4,
    )

    results_dir = _ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = results_dir / "best_model.pt"

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
    }

    best_val_f1 = -1.0
    epochs_no_improve = 0
    patience = 15
    max_epochs = 150

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(data.x, data.edge_index)
        logits_tr = out[train_indices]
        y_tr = data.y[train_indices]
        loss = criterion(logits_tr, y_tr)
        loss.backward()
        optimizer.step()
        history["train_loss"].append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            out_val = model(data.x, data.edge_index)
            vm = data.val_mask
            vloss = criterion(out_val[vm], data.y[vm]).item()
            pred = out_val[vm].argmax(dim=1).detach().cpu().numpy()
            true = data.y[vm].detach().cpu().numpy()
        vf1 = float(f1_score(true, pred, average="binary", zero_division=0))
        history["val_loss"].append(float(vloss))
        history["val_f1"].append(float(vf1))

        if vf1 > best_val_f1 + 1e-6:
            best_val_f1 = vf1
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    def _json_safe(h: dict[str, list[float]]) -> dict[str, list[float | None]]:
        def conv(v: float) -> float | None:
            if v != v or np.isinf(v):
                return None
            return v

        return {k: [conv(x) for x in vals] for k, vals in h.items()}

    with open(results_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(history), f, indent=2)

    return history
