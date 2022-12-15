from math import ceil
import torch
import torch_geometric.utils as pygu
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, aggr, GINEConv, SAGEConv, GATv2Conv, dense_diff_pool
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


def build_attentional_aggr(hidden_channels):
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
    return aggr.AttentionalAggregation(attn_gate_mlp, attn_mlp)


class DiffPool(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, poolevery=2, aggrtype='mean', aggrpool='mean', readout='mean', reduce_to=0.25):
        super().__init__()
        self.atom_encoder = AtomEncoder(emb_dim=hidden_channels)

        self.num_layers = num_layers
        self.poolevery = poolevery
        self.num_pool = (num_layers // poolevery) - 1
        self.drops = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.poolconv = nn.ModuleList()
        self.convassign = nn.ModuleList()
        self.norms = nn.ModuleList()

        clusters = 26  # mean num nodes in the molhiv dataset
        for l in range(num_layers + self.num_pool):
            if l > 0 and l % poolevery == 0 and len(self.poolconv) < self.num_pool:
                # pooling layer
                # reduce number of clusters
                clusters = int(ceil(clusters * reduce_to))
                assert clusters > 1

                # add placeholders to the list of convolutions
                self.convs.append(None)
                self.norms.append(None)
                self.drops.append(None)

                # add convolutions predicting the soft clustering assignments
                pooling_aggr = build_attentional_aggr(
                    hidden_channels) if aggrpool == 'attn' else aggrpool

                self.poolconv.append(
                    SAGEConv(hidden_channels, hidden_channels, aggr=pooling_aggr))
                self.convassign.append(
                    SAGEConv(hidden_channels, clusters, aggr=pooling_aggr))
            else:
                # graph convolution layer
                self.norms.append(nn.BatchNorm1d(hidden_channels))
                self.drops.append(nn.Dropout(p=0.5))

                # conv aggregation
                conv_aggr = build_attentional_aggr(
                    hidden_channels) if aggrtype == 'attn' else aggrtype
                self.convs.append(
                    SAGEConv(hidden_channels, hidden_channels, aggr=conv_aggr))

        # readout
        if readout == 'attn':
            self.readout = build_attentional_aggr(hidden_channels)
        elif readout == 'pool':
            # perform a final diffpool with a single cluster
            pooling_aggr = build_attentional_aggr(
                hidden_channels) if aggrpool == 'attn' else aggrpool
            self.readout = nn.ModuleList([
                SAGEConv(hidden_channels, hidden_channels, aggr=pooling_aggr),
                SAGEConv(hidden_channels, 1, aggr=pooling_aggr)
            ])
        elif readout == 'mean':
            self.readout = aggr.MeanAggregation()
        elif readout == 'max':
            self.readout = aggr.MaxAggregation()

        self.linear = nn.Linear(hidden_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def apply_diffpool(self, x, edge_index, s, batch_idx):
        # convert sparse batch representation to tradional dense batch
        x, _ = pygu.to_dense_batch(x, batch_idx)
        s, _ = pygu.to_dense_batch(s, batch_idx)
        edge_index = pygu.to_dense_adj(edge_index, batch_idx)

        # apply diffpool
        x, edge_index, link_pred_loss, entropy = dense_diff_pool(
            x, edge_index, s)

        # revert to sparse batching
        sparse_batch = Batch.from_data_list(
            [Data(x=x[b], edge_index=pygu.dense_to_sparse(edge_index[b])) for b in range(x.shape[0])])

        return sparse_batch.x, sparse_batch.edge_index[0], sparse_batch.batch, link_pred_loss, entropy

    def forward(self, x: torch.Tensor, edge_index, edge_feats=None, batch_idx=None):
        x = self.atom_encoder(x)

        link_losses = []
        assignments_losses = []

        pool_idx = 0
        for l in range(self.num_layers + self.num_pool):
            # None is a place holder in the layer list
            # if the current element is None, apply diffpool
            if self.convs[l] is not None:
                # conv layer
                res = x.clone()
                x = self.convs[l](x, edge_index)
                x = self.norms[l](x)
                # no relu on the last layer
                if l < self.num_layers-1:
                    x = x.relu()
                x = self.drops[l](x)
                # residual connection
                x += res
            else:
                # pooling layer
                # predict soft clustering assignments
                s = self.poolconv[pool_idx](x, edge_index)
                s = self.convassign[pool_idx](x, edge_index)

                x, edge_index, batch_idx, link_pred_loss, entropy = self.apply_diffpool(
                    x, edge_index, s, batch_idx)

                link_losses.append(link_pred_loss)
                assignments_losses.append(entropy)

                pool_idx += 1

        if isinstance(self.readout, nn.ModuleList):
            # pooling readout
            for conv in self.readout:
                s = conv(x, edge_index)
            x, edge_index, batch_idx, link_pred_loss, entropy = self.apply_diffpool(
                x, edge_index, s, batch_idx)
            link_losses.append(link_pred_loss)
            assignments_losses.append(entropy)
        else:
            # aggregation pooling
            x = self.readout(x, batch_idx)

        x = self.linear(x)
        x = self.sigmoid(x)
        return x, torch.Tensor(link_losses).to(x.device), torch.Tensor(assignments_losses).to(x.device)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, attnaggr=False, heads=1):
        super().__init__()
        self.atom_encoder = AtomEncoder(emb_dim=hidden_channels)
        self.bond_encoder = BondEncoder(emb_dim=hidden_channels)

        self.num_layers = num_layers
        self.drops = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(hidden_channels, hidden_channels // heads, edge_dim=hidden_channels, heads=heads))
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
            x = self.convs[l](
                x, edge_index, edge_attr=self.bond_encoder(edge_feats))
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


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, aggrtype='mean', readout='mean'):
        super().__init__()
        self.atom_encoder = AtomEncoder(emb_dim=hidden_channels)

        self.num_layers = num_layers
        self.drops = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.norms.append(nn.BatchNorm1d(hidden_channels))
            self.drops.append(nn.Dropout(p=0.5))

            # sage aggregation
            if aggrtype == 'attn':
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
            else:
                self.convs.append(
                    SAGEConv(hidden_channels, hidden_channels, aggr=aggrtype))

        # readout
        if readout == 'attn':
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
        elif readout == 'mean':
            self.readout = aggr.MeanAggregation()
        elif readout == 'max':
            self.readout = aggr.MaxAggregation()

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
