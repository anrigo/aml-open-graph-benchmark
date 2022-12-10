import torch
from torch_geometric.nn import GCNConv, aggr, GINConv
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()
        self.atom_encoder = AtomEncoder(emb_dim=in_channels)

        self.convs = nn.ModuleList(
            nn.Dropout(p=0.5),
            GCNConv(in_channels, hidden_channels)
        )

        for _ in range(num_layers-1):
            self.convs.append(nn.Dropout(p=0.5))
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.readout = aggr.MeanAggregation()
        self.linear = nn.Linear(hidden_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, edge_weight=None, batch_idx=None):
        x = self.atom_encoder(x)
        x = self.readout(x, batch_idx)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
