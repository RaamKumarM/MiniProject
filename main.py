"""End-to-end pipeline: load data → graph → train → evaluate → persist artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
for p in (_ROOT, _SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import config
from config import GAT_HEADS, GAT_HIDDEN_CHANNELS, GAT_OUT_CHANNELS
from data_loader import load_and_clean
from evaluate import evaluate_model
from graph_builder import build_graph
from model import DDoSGAT
from train import train_model


def _data_dir() -> Path:
    root = Path(config.__file__).resolve().parent
    return getattr(config, "DATA_DIR", root / "data" / "raw")


def _save_graph_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = data.num_nodes
    node_ips = getattr(data, "node_ips", None) or [str(i) for i in range(n)]
    nodes = [
        {"id": i, "label": node_ips[i], "attack": int(data.y[i].item())}
        for i in range(n)
    ]
    ei = data.edge_index.cpu().numpy()
    edges = [
        {"source": int(ei[0, j]), "target": int(ei[1, j])}
        for j in range(ei.shape[1])
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, indent=2)


def _json_safe_floats(vals: list[float]) -> list[float | None]:
    out: list[float | None] = []
    for v in vals:
        if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
            out.append(None)
        else:
            out.append(v)
    return out


def run_pipeline(csv_filename: str) -> dict:
    """Run load → graph → train → evaluate; persist graph, scaler, metrics, last dataset name."""
    safe_name = Path(csv_filename).name
    csv_path = _data_dir() / safe_name
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    results_dir = _ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] load_and_clean …", flush=True)
    df, scaler = load_and_clean([str(csv_path)])
    joblib.dump(scaler, results_dir / "scaler.pkl")

    print("[2/4] build_graph …", flush=True)
    data = build_graph(df)

    graph_path = results_dir / "graph_data.json"
    _save_graph_json(data, graph_path)
    print(f"      saved graph → {graph_path}", flush=True)

    print("[3/4] train_model …", flush=True)
    history = train_model(data)

    print("[4/4] evaluate_model …", flush=True)
    model = DDoSGAT(
        in_channels=data.x.shape[1],
        hidden_channels=GAT_HIDDEN_CHANNELS,
        out_channels=GAT_OUT_CHANNELS,
        heads=GAT_HEADS,
    )
    eval_metrics = evaluate_model(model, data)

    (results_dir / "last_dataset.txt").write_text(safe_name, encoding="utf-8")

    report = {
        **eval_metrics,
        "history": {
            "train_loss": history["train_loss"],
            "val_loss": _json_safe_floats(history["val_loss"]),
            "val_f1": _json_safe_floats(history["val_f1"]),
        },
    }
    with open(results_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Pipeline complete.", flush=True)

    return report
