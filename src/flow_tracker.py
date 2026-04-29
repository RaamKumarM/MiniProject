import time
import pandas as pd
import numpy as np
from scapy.all import IP, TCP, UDP
import sys
from pathlib import Path

# Add root to path for imports
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import FEATURE_COLUMNS

class FlowTracker:
    """Aggregates raw packets into flow-level summaries for GNN inference.
    
    COMPATIBILITY NOTE: 
    Training uses 78 features. Live capture calculates ~12 key features
    and maps them to the correct indices to satisfy the model/scaler shape.
    """
    
    def __init__(self):
        self.flows = {}
        # Mapping of (feature_name -> tracker_key) for supported live features
        self.feature_map = {
            "Destination Port": "dst_port",
            "Flow Duration": "duration",
            "Total Fwd Packets": "pkt_count",
            "Total Length of Fwd Packets": "byte_count",
            "Fwd Packet Length Max": "pkt_max",
            "Fwd Packet Length Min": "pkt_min",
            "Fwd Packet Length Mean": "pkt_mean",
            "Flow Bytes/s": "bytes_per_s",
            "Flow Packets/s": "pkts_per_s",
            "Average Packet Size": "avg_pkt_size",
            "SYN Flag Count": "syn_count",
            "ACK Flag Count": "ack_count"
        }

    def process_packet(self, pkt):
        if not pkt.haslayer(IP): return
        
        # Key: (src, dst)
        key = (pkt[IP].src, pkt[IP].dst)
        
        if key not in self.flows:
            self.flows[key] = {
                "src": key[0], "dst": key[1], "dst_port": 0,
                "duration": 0, "pkt_count": 0, "byte_count": 0,
                "pkt_max": 0, "pkt_min": 999999, "pkt_mean": 0,
                "syn_count": 0, "ack_count": 0,
                "first_seen": pkt.time, "last_seen": pkt.time
            }
        
        f = self.flows[key]
        f["pkt_count"] += 1
        size = len(pkt)
        f["byte_count"] += size
        f["pkt_max"] = max(f["pkt_max"], size)
        f["pkt_min"] = min(f["pkt_min"], size)
        f["last_seen"] = pkt.time
        
        if pkt.haslayer(TCP):
            f["dst_port"] = pkt[TCP].dport
            if pkt[TCP].flags & 0x02: f["syn_count"] += 1
            if pkt[TCP].flags & 0x10: f["ack_count"] += 1
        elif pkt.haslayer(UDP):
            f["dst_port"] = pkt[UDP].dport

    def flush(self):
        if not self.flows: return pd.DataFrame()

        rows = []
        for key, f in self.flows.items():
            duration = float(f["last_seen"] - f["first_seen"])
            f["duration"] = duration
            f["pkt_mean"] = f["byte_count"] / f["pkt_count"]
            f["bytes_per_s"] = f["byte_count"] / (duration + 0.001)
            f["pkts_per_s"] = f["pkt_count"] / (duration + 0.001)
            f["avg_pkt_size"] = f["pkt_mean"]

            # 1. Create empty 78-feature row
            row_data = {col: 0.0 for col in FEATURE_COLUMNS}
            
            # 2. Map calculated features to correct columns
            for feat_name, tracker_key in self.feature_map.items():
                if feat_name in row_data:
                    row_data[feat_name] = f[tracker_key]

            # 3. Add metadata for graph building
            row_data["Source IP"] = f["src"]
            row_data["Destination IP"] = f["dst"]
            row_data["Label"] = 0
            rows.append(row_data)

        self.flows = {}
        return pd.DataFrame(rows)
