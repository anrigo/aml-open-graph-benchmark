#%
# from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
# atom_encoder = AtomEncoder(emb_dim = 100)
# bond_encoder = BondEncoder(emb_dim = 100)

import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
from utils import visualize

# Download and process data at './dataset/ogbg_molhiv/'
dataset = PygGraphPropPredDataset(name = "ogbg-molhiv", root = 'dataset/')

 
split_idx = dataset.get_idx_split() 
train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)


batch = next(iter(train_loader))
sample = batch[0]
print(sample)
visualize(sample)
