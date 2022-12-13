import torch
from torch_geometric.nn import GCNConv, aggr, GINEConv, SAGEConv
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, attnaggr=False, shareattn=True):
        super().__init__()
        self.atom_encoder = AtomEncoder(emb_dim=hidden_channels)

        self.num_layers = num_layers
        self.drops = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.norms.append(nn.BatchNorm1d(hidden_channels))
            self.drops.append(nn.Dropout(p=0.5))

        if attnaggr:
            self.gate_mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, 2*hidden_channels),
                torch.nn.BatchNorm1d(2*hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(2*hidden_channels, 1),
                nn.Tanh()
            )
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, 2*hidden_channels),
                torch.nn.BatchNorm1d(2*hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(2*hidden_channels, hidden_channels)
            )
            for _ in range(num_layers):
                if shareattn:
                    self.convs.append(SAGEConv(hidden_channels, hidden_channels,
                                               aggr=aggr.AttentionalAggregation(self.gate_mlp, self.mlp)))
                else:
                    attn_gate_mlp = torch.nn.Sequential(
                        torch.nn.Linear(hidden_channels, 2*hidden_channels),
                        torch.nn.BatchNorm1d(2*hidden_channels),
                        torch.nn.ReLU(),
                        torch.nn.Linear(2*hidden_channels, 1),
                        nn.Tanh()
                    )
                    attn_mlp = torch.nn.Sequential(
                        torch.nn.Linear(hidden_channels, 2*hidden_channels),
                        torch.nn.BatchNorm1d(2*hidden_channels),
                        torch.nn.ReLU(),
                        torch.nn.Linear(2*hidden_channels, hidden_channels)
                    )
                    self.convs.append(SAGEConv(hidden_channels, hidden_channels,
                                               aggr=aggr.AttentionalAggregation(attn_gate_mlp, attn_mlp)))

            self.readout = aggr.AttentionalAggregation(self.gate_mlp, self.mlp)
        else:
            for _ in range(num_layers):
                self.convs.append(
                    SAGEConv(hidden_channels, hidden_channels, aggr='mean'))
            self.readout = aggr.MeanAggregation()

        self.linear = nn.Linear(hidden_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, edge_index, edge_feats=None, batch_idx=None):
        x = self.atom_encoder(x)

        for l in range(self.num_layers):
            res = x.clone()
            x = self.convs[l](x, edge_index)
            x = self.norms[l](x)
            # no relu on the last layer
            if l < self.num_layers-1:
                x = x.relu()
            x = self.drops[l](x)
            # residual connection
            x += res

        x = self.readout(x, batch_idx)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, attnaggr=False):
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

        if attnaggr:
            self.gate_mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, 2*hidden_channels),
                torch.nn.BatchNorm1d(2*hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(2*hidden_channels, 1),
                nn.Tanh()
            )
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, 2*hidden_channels),
                torch.nn.BatchNorm1d(2*hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(2*hidden_channels, hidden_channels)
            )
            self.readout = aggr.AttentionalAggregation(self.gate_mlp, self.mlp)
        else:
            self.readout = aggr.MeanAggregation()

        self.linear = nn.Linear(hidden_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, edge_index, edge_feats=None, batch_idx=None):
        x = self.atom_encoder(x)

        for l in range(self.num_layers):
            res = x.clone()
            x = self.convs[l](x, edge_index)
            x = self.norms[l](x)
            # no relu on the last layer
            if l < self.num_layers-1:
                x = x.relu()
            x = self.drops[l](x)
            # residual connection
            x += res

        x = self.readout(x, batch_idx)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


class EdgeGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, attnaggr=False):
        super().__init__()
        self.atom_encoder = AtomEncoder(emb_dim=hidden_channels)
        self.bond_encoder = BondEncoder(emb_dim=hidden_channels)

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, 2*hidden_channels),
            torch.nn.BatchNorm1d(2*hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(2*hidden_channels, 1),
            nn.Tanh()
        )

        self.num_layers = num_layers
        self.drops = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(nn.BatchNorm1d(hidden_channels))
            self.drops.append(nn.Dropout(p=0.5))

        if attnaggr:
            self.gate_mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, 2*hidden_channels),
                torch.nn.BatchNorm1d(2*hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(2*hidden_channels, 1),
                nn.Tanh()
            )
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, 2*hidden_channels),
                torch.nn.BatchNorm1d(2*hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(2*hidden_channels, hidden_channels)
            )
            self.readout = aggr.AttentionalAggregation(self.gate_mlp, self.mlp)
        else:
            self.readout = aggr.MeanAggregation()

        self.linear = nn.Linear(hidden_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, edge_index, edge_feats=None, batch_idx=None):
        x = self.atom_encoder(x)
        # predict weights in [1,3] from edge features
        edge_weights = self.regressor(self.bond_encoder(edge_feats)) + 2

        for l in range(self.num_layers):
            res = x.clone()
            x = self.convs[l](x, edge_index, edge_weights)
            x = self.norms[l](x)
            # no relu on the last layer
            if l < self.num_layers-1:
                x = x.relu()
            x = self.drops[l](x)
            # residual connection
            x += res

        x = self.readout(x, batch_idx)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


class GINE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()
        self.atom_encoder = AtomEncoder(emb_dim=hidden_channels)
        self.bond_encoder = BondEncoder(emb_dim=hidden_channels)

        self.num_layers = num_layers
        self.drops = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(GINEConv(
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_channels, 2*hidden_channels), torch.nn.BatchNorm1d(
                        2*hidden_channels), torch.nn.ReLU(), torch.nn.Linear(2*hidden_channels, hidden_channels)
                ),
                hidden_channels, hidden_channels
            ))
            self.norms.append(nn.BatchNorm1d(hidden_channels))
            self.drops.append(nn.Dropout(p=0.5))

        self.readout = aggr.MeanAggregation()
        self.linear = nn.Linear(hidden_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, edge_index, edge_feats=None, batch_idx=None):
        x = self.atom_encoder(x)
        edge_feats = self.bond_encoder(edge_feats)

        for l in range(self.num_layers):
            res = x.clone()
            x = self.convs[l](x, edge_index, edge_feats)
            x = self.norms[l](x)
            # no relu on the last layer
            if l < self.num_layers-1:
                x = x.relu()
            x = self.drops[l](x)
            # residual connection
            x += res

        x = self.readout(x, batch_idx)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
