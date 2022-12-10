import torch
from torch_geometric.nn import GCNConv, aggr, GINConv
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()
        self.atom_encoder = AtomEncoder(emb_dim=hidden_channels)

        self.num_layers = num_layers
        self.drops = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(nn.BatchNorm1d(hidden_channels))
            self.drops.append(nn.Dropout(p=0.5))

        self.readout = aggr.MeanAggregation()
        self.linear = nn.Linear(hidden_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, edge_index, edge_weight=None, batch_idx=None):
        x = self.atom_encoder(x)

        for l in range(self.num_layers):
            res = x.clone()
            x = self.convs[l](x, edge_index).relu()
            x = self.norms[l](x)
            # no relu on the last layer
            if l < self.num_layers:
                x = x.relu()
            x = self.drops[l](x)
            # residual connection
            x += res

        x = self.readout(x, batch_idx)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
