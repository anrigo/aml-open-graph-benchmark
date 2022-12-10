import torch
from torch_geometric.nn import GCNConv, aggr, GINConv
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class SimpleGIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.atom_encoder = AtomEncoder(emb_dim=in_channels)
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 2*hidden_channels),
            torch.nn.BatchNorm1d(2*hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(2*hidden_channels, hidden_channels)
        )
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, 2*hidden_channels),
            torch.nn.BatchNorm1d(2*hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(2*hidden_channels, hidden_channels)
        )
        self.mlp3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, 2*hidden_channels),
            torch.nn.BatchNorm1d(2*hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(2*hidden_channels, hidden_channels)
        )
        self.drop1 = nn.Dropout(p=0.5)
        self.conv1 = GINConv(self.mlp1, in_channels, hidden_channels)
        self.drop2 = nn.Dropout(p=0.5)
        self.conv2 = GINConv(self.mlp2, hidden_channels, hidden_channels)
        self.drop3 = nn.Dropout(p=0.5)
        self.conv3 = GINConv(self.mlp3, hidden_channels, hidden_channels)
        self.readout = aggr.MeanAggregation()
        self.linear = nn.Linear(hidden_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, edge_weight=None, batch_idx=None):
        x = self.atom_encoder(x)
        x = self.drop1(x)
        x = self.conv1(x, edge_index).relu()
        x = self.drop2(x)
        x = self.conv2(x, edge_index).relu()
        x = self.drop3(x)
        x = self.conv3(x, edge_index).relu()
        x = self.readout(x, batch_idx)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


class SimpleGCN(torch.nn.Module):
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
        x = self.conv2(x, edge_index).relu()
        x = self.drop3(x)
        x = self.conv3(x, edge_index).relu()
        x = self.readout(x, batch_idx)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


class EdgeGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.atom_encoder = AtomEncoder(emb_dim=in_channels)
        self.to_edge_weight = nn.Linear(3, 1)
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
        # project edge features to edge weights > 1
        edge_weight = self.to_edge_weight(edge_weight).relu() + 1
        x = self.drop1(x)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.drop2(x)
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = self.drop3(x)
        x = self.conv3(x, edge_index, edge_weight).relu()
        x = self.readout(x, batch_idx)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
