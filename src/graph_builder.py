from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import FEATURE_COLUMNS

_LABEL = "Label"


def _resolve_ip_columns(df: pd.DataFrame) -> tuple[str, str]:
    for src, dst in (
        ("Source IP", "Destination IP"),
        ("Src IP", "Dst IP"),
    ):
        if src in df.columns and dst in df.columns:
            return src, dst
    raise ValueError(
        "DataFrame must contain IP columns ('Source IP'/'Destination IP' "
        "or 'Src IP'/'Dst IP')."
    )


def build_graph(df: pd.DataFrame, is_training: bool = True) -> Data:
    """Build a PyG graph: nodes are unique IPs; edges are flows (src → dst)."""
    src_col, dst_col = _resolve_ip_columns(df)

    feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not feat_cols:
        raise ValueError("No FEATURE_COLUMNS from config are present in the DataFrame.")

    # In live inference, we might not have a Label column.
    has_label = _LABEL in df.columns

    src_ips = df[src_col].astype(str)
    dst_ips = df[dst_col].astype(str)
    unique_ips = pd.unique(pd.concat([src_ips, dst_ips], ignore_index=True))
    ip_to_id = {ip: i for i, ip in enumerate(unique_ips)}
    num_nodes = len(ip_to_id)

    src_ids = src_ips.map(ip_to_id).to_numpy(dtype=np.int64)
    dst_ids = dst_ips.map(ip_to_id).to_numpy(dtype=np.int64)
    edge_index = torch.tensor(np.stack([src_ids, dst_ids], axis=0), dtype=torch.long)

    feat_block = df[feat_cols].to_numpy(dtype=np.float64)
    feat_long = np.vstack([feat_block, feat_block])
    long_nids = np.concatenate([src_ids, dst_ids])
    
    # Track labels if available
    if has_label:
        labels = df[_LABEL].to_numpy()
        long_labels = np.concatenate([labels, labels])
    else:
        long_labels = np.zeros(len(long_nids), dtype=np.int64)

    long_df = pd.DataFrame(feat_long, columns=feat_cols, index=long_nids)
    long_df[_LABEL] = long_labels

    gb = long_df.groupby(level=0)
    x_agg = gb[feat_cols].mean()
    y_agg = gb[_LABEL].max()

    x_np = x_agg.reindex(range(num_nodes)).fillna(0).to_numpy(dtype=np.float64)
    y_np = y_agg.reindex(range(num_nodes)).fillna(0).to_numpy(dtype=np.int64)

    x = torch.tensor(x_np, dtype=torch.float)
    y = torch.tensor(y_np, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)

    if is_training:
        rng = np.random.default_rng(42)
        perm = rng.permutation(num_nodes)
        n_train = int(0.70 * num_nodes)
        n_val = int(0.15 * num_nodes)
        i_train = perm[:n_train]
        i_val = perm[n_train : n_train + n_val]
        i_test = perm[n_train + n_val :]

        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[i_train] = True
        data.val_mask[i_val] = True
        data.test_mask[i_test] = True

    # One string label per node id (same order as ip_to_id / enumerate(unique_ips))
    data.node_ips = [str(unique_ips[i]) for i in range(num_nodes)]
    return data
