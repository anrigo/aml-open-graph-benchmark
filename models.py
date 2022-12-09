import torch
from torch_geometric.nn import GCNConv, aggr
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.atom_encoder = AtomEncoder(emb_dim=in_channels)
        self.drop1 = nn.Dropout(p=0.5)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.drop2 = nn.Dropout(p=0.5)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.drop3 = nn.Dropout(p=0.5)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.readout = aggr.MeanAggregation()
        self.linear = nn.Linear(hidden_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, edge_weight=None, batch_idx=None):
        x = self.atom_encoder(x)
        x = self.drop1(x)
        x = self.conv1(x, edge_index).relu()
        x = self.drop2(x)
        x = self.conv2(x, edge_index)
        x = self.readout(x, batch_idx)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x