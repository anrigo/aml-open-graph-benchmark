import argparse
import os
from pathlib import Path
import torch
from tqdm import tqdm
from utils import accuracy
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader
import models


@torch.no_grad()
def eval(model, loader, evaluator, split=None, device=None):
    '''Evaluate the model'''

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    desc = 'Evaluation' if split is None else f'Evaluating on {split}'
    if split is None:
        split = ''
    else:
        split += '_'

    model.to(device)
    model.eval()
    y_true = []
    y_pred = []

    for _, batch in enumerate(tqdm(loader, desc=desc)):
        batch = batch.to(device)

        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    acc = accuracy(y_true.clone(), y_pred.clone())

    input_dict = {"y_true": y_true.numpy(), "y_pred": y_pred.numpy()}
    rocauc_dict = evaluator.eval(input_dict)

    return {f'{split}accuracy': acc, f'{split}rocauc': rocauc_dict['rocauc']}


def test(args):
    rundir = Path('runs', args.run)

    if not rundir.exists():
        raise FileNotFoundError('Run folder not found')
    
    checkpoints = [file_ for file_ in os.listdir(rundir) if '.pth' in file_]

    if len(checkpoints) == 0:
        raise FileNotFoundError('No checkpoint found')
    
    device = torch.device(
        'cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')

    # Download and process data at './dataset/ogbg_molhiv/'
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root='dataset/')

    split_idx = dataset.get_idx_split()
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False)
    
    evaluator = Evaluator(name="ogbg-molhiv")

    for chk in checkpoints:
        print(f'Evaluating {chk}')
        model = models.GCN(dataset.num_features, args.emb_dim,
                           args.layers, attnaggr=False).to(device)

        state_dict = torch.load(Path(rundir, chk))
        model.load_state_dict(state_dict)

        metrics = eval(model, test_loader, evaluator, 'test', device)
        print(metrics)


if __name__ == '__main__':
    # Evaluation settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int,
                        default=64, help='batch size')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--layers', type=int, default=6,
                        help='number of GNN layers (default: 6)')
    parser.add_argument('--cpu', action=argparse.BooleanOptionalAction,
                        default=False, help='run on cpu only')
    parser.add_argument('--run', type=str, help='run name')

    args = parser.parse_args()

    test(args)
