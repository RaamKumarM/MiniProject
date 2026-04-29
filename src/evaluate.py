from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch_geometric.data import Data

_ROOT = Path(__file__).resolve().parent.parent
_SRC = Path(__file__).resolve().parent
for p in (_ROOT, _SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load_training_history(path: Path) -> dict[str, list[float | None]] | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _plot_training_curves(
    history: dict[str, list[float | None]] | None,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if not history or "train_loss" not in history:
        ax.text(0.5, 0.5, "No training history", ha="center", va="center")
        ax.set_axis_off()
    else:
        train_loss = np.array(history["train_loss"], dtype=float)
        epochs = np.arange(1, len(train_loss) + 1)
        ax.plot(epochs, train_loss, label="train loss")

        val_loss = history.get("val_loss")
        if val_loss is not None:
            v = np.array(val_loss, dtype=float)
            mask = ~np.isnan(v)
            if np.any(mask):
                ax.plot(epochs[mask], v[mask], label="val loss")

        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("Training curves")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_model(model: nn.Module, data: Data) -> dict[str, float | list[list[int]]]:
    """Load best weights, evaluate on test set, save plots, return metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    model = model.to(device)

    results_dir = _ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = results_dir / "best_model.pt"
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        mask = data.test_mask
        logits = out[mask]
        pred = logits.argmax(dim=1).detach().cpu().numpy()
        true = data.y[mask].detach().cpu().numpy()

    print(
        classification_report(
            true,
            pred,
            labels=[0, 1],
            target_names=["benign", "attack"],
            digits=4,
            zero_division=0,
        ),
        flush=True,
    )

    acc = float(accuracy_score(true, pred))
    prec = float(precision_score(true, pred, average="macro", zero_division=0))
    rec = float(recall_score(true, pred, average="macro", zero_division=0))
    f1 = float(f1_score(true, pred, average="macro", zero_division=0))
    cm = confusion_matrix(true, pred, labels=[0, 1])
    cm_list = cm.tolist()

    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["pred 0", "pred 1"],
        yticklabels=["true 0", "true 1"],
        ax=ax_cm,
    )
    ax_cm.set_title("Confusion matrix (test)")
    fig_cm.tight_layout()
    fig_cm.savefig(results_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig_cm)

    history = _load_training_history(results_dir / "training_history.json")
    _plot_training_curves(history, results_dir / "training_curves.png")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm_list,
    }
