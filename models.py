import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.atom_encoder = AtomEncoder(emb_dim=in_channels)
        self.bond_encoder = BondEncoder(emb_dim=1)
        self.drop1 = nn.Dropout(p=0.5)
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.drop2 = nn.Dropout(p=0.5)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True)
        self.linear = nn.Linear(out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.atom_encoder(x)
        edge_weight = self.bond_encoder(edge_weight)
        x = self.drop1(x)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.drop2(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
