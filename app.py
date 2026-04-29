"""Flask backend for DDoS GNN detection: training, results, network stats, live SSE."""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from flask import Flask, Response, jsonify, render_template, request, send_file

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
for p in (_ROOT, _SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import (
    FEATURE_COLUMNS,
    GAT_HEADS,
    GAT_HIDDEN_CHANNELS,
    GAT_OUT_CHANNELS,
    STREAM_CHUNK_SIZE,
    STREAM_INTERVAL,
    STREAM_OVERLAP,
)
from graph_builder import build_graph
from main import run_pipeline
from model import DDoSGAT

app = Flask(__name__, template_folder="templates")

RESULTS_DIR = _ROOT / "results"
CKPT_PATH = RESULTS_DIR / "best_model.pt"
SCALER_PATH = RESULTS_DIR / "scaler.pkl"
LAST_DATASET_PATH = RESULTS_DIR / "last_dataset.txt"
EVAL_JSON_PATH = RESULTS_DIR / "evaluation_metrics.json"
GRAPH_JSON_PATH = RESULTS_DIR / "graph_data.json"
CM_PATH = RESULTS_DIR / "confusion_matrix.png"
RAW_DIR = _ROOT / "data" / "raw"

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL: DDoSGAT | None = None
SCALER = None
TRAIN_FILE: str | None = None
DETECTION_FILE: str | None = None
LIVE_RUNNING = False


def _load_ckpt_state(path: Path) -> dict:
    try:
        return torch.load(path, map_location=_device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=_device)


def _infer_in_channels(state_dict: dict) -> int:
    for key in ("conv1.lin_src.weight", "conv1.lin.weight"):
        t = state_dict.get(key)
        if t is not None and hasattr(t, "ndim") and t.ndim == 2:
            return int(t.shape[1])
    for k, v in state_dict.items():
        if (
            k.startswith("conv1.")
            and "lin" in k
            and hasattr(v, "ndim")
            and v.ndim == 2
        ):
            return int(v.shape[1])
    raise ValueError("Cannot infer input channels from checkpoint.")


def _read_train_file() -> str | None:
    if not LAST_DATASET_PATH.is_file():
        return None
    name = LAST_DATASET_PATH.read_text(encoding="utf-8").strip()
    return name or None


def load_globals_at_startup() -> None:
    global MODEL, SCALER, TRAIN_FILE, DETECTION_FILE
    TRAIN_FILE = _read_train_file()
    DETECTION_FILE = TRAIN_FILE

    if SCALER_PATH.is_file():
        try:
            SCALER = joblib.load(SCALER_PATH)
        except Exception:
            SCALER = None
    else:
        SCALER = None

    if CKPT_PATH.is_file():
        try:
            state = _load_ckpt_state(CKPT_PATH)
            in_ch = _infer_in_channels(state)
            m = DDoSGAT(
                in_channels=in_ch,
                hidden_channels=GAT_HIDDEN_CHANNELS,
                out_channels=GAT_OUT_CHANNELS,
                heads=GAT_HEADS,
            ).to(_device)
            m.load_state_dict(state)
            m.eval()
            MODEL = m
        except Exception:
            MODEL = None
    else:
        MODEL = None


def _reload_model_scaler() -> None:
    global MODEL, SCALER, TRAIN_FILE, DETECTION_FILE
    TRAIN_FILE = _read_train_file()
    if DETECTION_FILE is None:
        DETECTION_FILE = TRAIN_FILE

    if SCALER_PATH.is_file():
        try:
            SCALER = joblib.load(SCALER_PATH)
        except Exception:
            SCALER = None
    else:
        SCALER = None

    if CKPT_PATH.is_file():
        try:
            state = _load_ckpt_state(CKPT_PATH)
            in_ch = _infer_in_channels(state)
            m = DDoSGAT(
                in_channels=in_ch,
                hidden_channels=GAT_HIDDEN_CHANNELS,
                out_channels=GAT_OUT_CHANNELS,
                heads=GAT_HEADS,
            ).to(_device)
            m.load_state_dict(state)
            m.eval()
            MODEL = m
        except Exception:
            MODEL = None
    else:
        MODEL = None


def _resolve_ip_columns(df: pd.DataFrame) -> tuple[str, str]:
    for src, dst in (
        ("Source IP", "Destination IP"),
        ("Src IP", "Dst IP"),
    ):
        if src in df.columns and dst in df.columns:
            return src, dst
    raise ValueError(
        "CSV must contain Source IP/Destination IP or Src IP/Dst IP columns."
    )


def _prepare_chunk_for_inference(chunk: pd.DataFrame, scaler) -> pd.DataFrame | None:
    df = chunk.copy()
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if "Label" not in df.columns:
        df["Label"] = 0
    else:
        lab = df["Label"].astype(str).str.strip()
        df["Label"] = np.where(lab.str.casefold() == "benign", 0, 1)

    present = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not present:
        return None

    src_col, dst_col = _resolve_ip_columns(df)
    df.dropna(subset=present + [src_col, dst_col, "Label"], inplace=True)
    if len(df) == 0:
        return None

    for col in present:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=present, inplace=True)
    if len(df) == 0:
        return None

    # Pass as DataFrame to preserve feature names and avoid sklearn warnings
    df[present] = scaler.transform(df[present])
    return df


load_globals_at_startup()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/list-datasets")
def list_datasets():
    names = sorted(p.name for p in RAW_DIR.glob("*.csv") if p.is_file())
    return jsonify({"files": names})


@app.route("/train", methods=["POST"])
def train():
    body = request.get_json(silent=True) or {}
    filename = body.get("filename")
    if not filename or not isinstance(filename, str):
        return jsonify({"error": "JSON body must include 'filename'."}), 400

    try:
        metrics = run_pipeline(Path(filename).name)
        _reload_model_scaler()
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/set-detection-dataset", methods=["POST"])
def set_detection_dataset():
    global DETECTION_FILE
    body = request.get_json(silent=True) or {}
    filename = body.get("filename")
    if not filename or not isinstance(filename, str):
        return jsonify({"error": "JSON body must include 'filename'."}), 400
    safe = Path(filename).name
    if not (RAW_DIR / safe).is_file():
        return jsonify({"error": f"File not found: data/raw/{safe}"}), 400
    DETECTION_FILE = safe
    return jsonify({"ok": True})


@app.route("/results")
def results():
    if not EVAL_JSON_PATH.is_file():
        return jsonify({"error": "not found"}), 404
    with open(EVAL_JSON_PATH, encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.route("/results/confusion_matrix.png")
def confusion_matrix_png():
    if not CM_PATH.is_file():
        return jsonify({"error": "not found"}), 404
    return send_file(CM_PATH, mimetype="image/png")


@app.route("/network-stats")
def network_stats():
    if not GRAPH_JSON_PATH.is_file():
        return jsonify({"error": "not found"}), 404
    try:
        with open(GRAPH_JSON_PATH, encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return jsonify({"error": str(e)}), 500

    nodes = payload.get("nodes") or []
    edges = payload.get("edges") or []

    total_nodes = len(nodes)
    total_edges = len(edges)

    id_to_ip = {int(n["id"]): str(n.get("label", n["id"])) for n in nodes}
    degree: dict[int, int] = defaultdict(int)
    for n in nodes:
        degree[int(n["id"])] = 0
    for e in edges:
        s, t = int(e["source"]), int(e["target"])
        degree[s] += 1
        degree[t] += 1

    ranked = sorted(degree.items(), key=lambda x: (-x[1], x[0]))[:10]
    top_connected = [
        {"ip": id_to_ip.get(nid, str(nid)), "connections": cnt}
        for nid, cnt in ranked
    ]

    attack_nodes = sum(1 for n in nodes if int(n.get("attack", 0)) == 1)
    benign_nodes = sum(1 for n in nodes if int(n.get("attack", 0)) != 1)

    return jsonify(
        {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "attack_nodes": attack_nodes,
            "benign_nodes": benign_nodes,
            "top_connected": top_connected,
        }
    )


@app.route("/live-stream")
def live_stream():
    """Near-real-time streaming inference using sliding windows."""
    global LIVE_RUNNING
    LIVE_RUNNING = True

    def generate():
        global LIVE_RUNNING
        if MODEL is None or SCALER is None:
            yield f"data: {json.dumps({'error': 'No trained model/scaler found. Train first.'})}\n\n"
            return

        stream_name = DETECTION_FILE if DETECTION_FILE else TRAIN_FILE
        if not stream_name:
            yield f"data: {json.dumps({'error': 'No dataset selected.'})}\n\n"
            return

        csv_path = RAW_DIR / Path(stream_name).name
        if not csv_path.is_file():
            yield f"data: {json.dumps({'error': f'File not found: {csv_path.name}'})}\n\n"
            return

        model, scaler = MODEL, SCALER
        
        # Buffer for sliding window
        overlap_buffer = pd.DataFrame()
        chunk_id = 0

        try:
            # We read STREAM_CHUNK_SIZE at a time
            reader = pd.read_csv(
                csv_path,
                chunksize=STREAM_CHUNK_SIZE,
                on_bad_lines="skip",
                low_memory=False,
                encoding="utf-8",
            )
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        for chunk in reader:
            if not LIVE_RUNNING:
                break

            # 1. Sliding Window: Combine current chunk with overlap from previous
            full_window = pd.concat([overlap_buffer, chunk], ignore_index=True)
            
            # Save the end of this window to be the 'overlap' for the next batch
            overlap_buffer = chunk.tail(STREAM_OVERLAP).copy()

            # 2. Preprocess
            prepared = _prepare_chunk_for_inference(full_window, scaler)
            if prepared is None or len(prepared) == 0:
                continue

            # 3. Build Graph (Inference mode)
            try:
                data = build_graph(prepared, is_training=False)
            except Exception as e:
                continue

            # 4. Model Prediction
            x = data.x.to(_device)
            edge_index = data.edge_index.to(_device)
            
            with torch.no_grad():
                logits = model(x, edge_index)
                prob = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1).cpu().numpy()
                prob_cpu = prob.cpu().numpy()

            # 5. Map Predictions back to Source IPs and Emit Events
            # To restore compatibility with the frontend, we emit one event per flow
            node_ips = data.node_ips
            ip_to_nid = {str(node_ips[i]): i for i in range(len(node_ips))}
            
            src_col, dst_col = _resolve_ip_columns(prepared)

            for _, row in prepared.iterrows():
                if not LIVE_RUNNING:
                    break
                src_ip = str(row.get(src_col, ""))
                dst_ip = str(row.get(dst_col, ""))
                nid = ip_to_nid.get(src_ip)
                if nid is None:
                    continue
                
                pred = int(preds[nid])
                conf = float(prob_cpu[nid, pred])
                alert = (pred == 1)
                
                payload = {
                    "timestamp": time.strftime("%H:%M:%S"),
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "prediction": pred,
                    "confidence": conf,
                    "alert": alert
                }
                yield f"data: {json.dumps(payload)}\n\n"
            
            chunk_id += 1
            # Delay between batches to simulate real-time
            time.sleep(STREAM_INTERVAL)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/live-packets")
def live_packets():
    """Endpoint for real-time packet capture and GNN detection."""
    from src.live_capture import LiveSniffer
    
    if MODEL is None or SCALER is None:
        return jsonify({"error": "No trained model found. Please train first."}), 400

    # Pass the local _prepare_chunk_for_inference to avoid circular imports
    sniffer = LiveSniffer(MODEL, SCALER, _device, _prepare_chunk_for_inference)
    sniffer.start()

    def generate():
        try:
            while sniffer.running:
                res = sniffer.get_window_prediction()
                if res:
                    yield f"data: {json.dumps(res)}\n\n"
                else:
                    time.sleep(1)
        finally:
            sniffer.stop()

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/live-stop", methods=["POST"])
def live_stop():
    global LIVE_RUNNING
    LIVE_RUNNING = False
    return jsonify({"ok": True})


@app.route("/shutdown", methods=["POST"])
def shutdown():
    def _kill() -> None:
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=_kill, daemon=True).start()
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, threaded=True, use_reloader=False)
