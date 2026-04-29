from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import GAT_DROPOUT


class DDoSGAT(nn.Module):
    """Two-layer GAT for binary node classification (logits over two classes)."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=True,
            dropout=0.0,
        )
        self.conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=True,
            dropout=0.0,
        )
        self.dropout = nn.Dropout(GAT_DROPOUT)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x
