"""Project configuration for ddos_gnn."""

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent

# CSV paths for the pipeline (default: all files under data/raw/).
DATA_FILES = [str(p) for p in sorted((_PROJECT_ROOT / "data" / "raw").glob("*.csv"))]

# Numerical flow features in CIC-IDS2019 CSVs (CICFlowMeter); excludes the Label column.
FEATURE_COLUMNS = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size",
    "Avg Bwd Segment Size", "Fwd Header Length.1", "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets",
    "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd",
    "min_seg_size_forward", "Active Mean", "Active Std", "Active Max",
    "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

# GAT Hyperparameters
GAT_DROPOUT = 0.3
GAT_HIDDEN_CHANNELS = 64
GAT_HEADS = 4
GAT_OUT_CHANNELS = 2

# Training
TRAIN_LR = 1e-3
TRAIN_EPOCHS = 150
VAL_EVERY = 10

# Streaming / Simulated Live Inference
STREAM_CHUNK_SIZE = 1000
STREAM_OVERLAP = 200
STREAM_INTERVAL = 1.0 

# REAL Live Packet Capture (Scapy)
# Windows Loopback: "\\Device\\NPF_Loopback"
# Linux: "eth0" or "wlan0"
LIVE_INTERFACE = "\\Device\\NPF_Loopback" 
LIVE_SNIFF_FILTER = "ip"
LIVE_WINDOW_DURATION = 3.0
LIVE_MAX_PACKETS = 5000
