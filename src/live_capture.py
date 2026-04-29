import json
import time
import torch
import pandas as pd
from scapy.all import sniff
from threading import Thread, Lock
from pathlib import Path
import sys

# Add root to path for imports
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    LIVE_INTERFACE, 
    LIVE_SNIFF_FILTER, 
    LIVE_WINDOW_DURATION, 
    LIVE_MAX_PACKETS
)
from src.flow_tracker import FlowTracker
from src.graph_builder import build_graph

class LiveSniffer:
    """Manages background packet sniffing and real-time GNN inference windows."""
    
    def __init__(self, model, scaler, device, prepare_fn):
        self.model = model
        self.scaler = scaler
        self.device = device
        self.prepare_fn = prepare_fn # Pass function from app to avoid circular import
        self.tracker = FlowTracker()
        self.lock = Lock()
        self.running = False
        self.pkt_count = 0

    def _packet_callback(self, pkt):
        with self.lock:
            if self.pkt_count < LIVE_MAX_PACKETS:
                self.tracker.process_packet(pkt)
                self.pkt_count += 1

    def start(self):
        """Starts the sniffing thread."""
        if self.running: return
        self.running = True
        self.sniff_thread = Thread(target=self._run_sniffer, daemon=True)
        self.sniff_thread.start()

    def _run_sniffer(self):
        print(f"[*] Sniffing on {LIVE_INTERFACE}...")
        try:
            sniff(
                iface=LIVE_INTERFACE,
                filter=LIVE_SNIFF_FILTER,
                prn=self._packet_callback,
                store=False,
                stop_filter=lambda x: not self.running
            )
        except Exception as e:
            print(f"[!] Sniffer Error: {e}")
            self.running = False

    def stop(self):
        self.running = False

    def get_window_prediction(self):
        """Processes the current window of packets and returns detection results."""
        # Wait for the window to fill
        time.sleep(LIVE_WINDOW_DURATION)
        
        with self.lock:
            df = self.tracker.flush()
            total_pkts = self.pkt_count
            self.pkt_count = 0

        if df.empty:
            return {
                "timestamp": time.strftime("%H:%M:%S"),
                "total_packets": total_pkts,
                "status": "Idle",
                "attack_count": 0
            }

        # 1. Scale features using passed prepare function
        prepared = self.prepare_fn(df, self.scaler)
        
        if prepared is None or len(prepared) == 0:
            return None

        # 2. Build graph (inference mode)
        data = build_graph(prepared, is_training=False)
        x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)
        
        # 3. Model Inference
        with torch.no_grad():
            logits = self.model(x, edge_index)
            preds = logits.argmax(dim=1).cpu().numpy()
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        # 4. Extract suspicious IPs and All Flows
        attack_indices = [i for i, p in enumerate(preds) if p == 1]
        suspicious = []
        for idx in attack_indices:
            suspicious.append({
                "ip": data.node_ips[idx],
                "confidence": float(probs[idx, 1])
            })

        # Get all flows for real-time visualization
        all_flows = []
        # We use prepared DF to get IPs and match with predictions
        # Note: prepared should have 'Source IP' and 'Destination IP' columns as per flush()
        src_col = "Source IP"
        dst_col = "Destination IP"
        
        # Mapping IP to its prediction from the graph nodes
        ip_to_pred = {data.node_ips[i]: int(preds[i]) for i in range(len(data.node_ips))}
        ip_to_conf = {data.node_ips[i]: float(probs[i, preds[i]]) for i in range(len(data.node_ips))}

        for _, row in prepared.iterrows():
            s_ip = str(row.get(src_col, ""))
            d_ip = str(row.get(dst_col, ""))
            p = ip_to_pred.get(s_ip, 0)
            c = ip_to_conf.get(s_ip, 0.0)
            
            all_flows.append({
                "src_ip": s_ip,
                "dst_ip": d_ip,
                "prediction": p,
                "confidence": c,
                "alert": (p == 1)
            })

        return {
            "timestamp": time.strftime("%H:%M:%S"),
            "total_packets": total_pkts,
            "total_flows": len(df),
            "attack_count": len(attack_indices),
            "status": "DDoS Detected" if attack_indices else "Normal",
            "suspicious_ips": suspicious[:10],
            "all_flows": all_flows[:50] # Limit to 50 flows per window for performance
        }
