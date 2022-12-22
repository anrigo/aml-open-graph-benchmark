import argparse
import os
from pathlib import Path
import torch
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import models
from utils import accuracy
from pandas import DataFrame
from copy import deepcopy


@torch.no_grad()
def eval(model, loader, evaluator, split=None, device=None, progress=True):
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

    teval = tqdm(loader, desc=desc, unit='batch') if progress else loader

    for batch in teval:
        batch = batch.to(device)

        if isinstance(model, models.DiffPool):
            pred, *_ = model(batch.x, batch.edge_index,
                             batch.edge_attr, batch.batch)
        else:
            pred = model(batch.x, batch.edge_index,
                         batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    acc = accuracy(y_true.clone(), y_pred.clone())

    input_dict = {"y_true": y_true.numpy(), "y_pred": y_pred.numpy()}
    rocauc_dict = evaluator.eval(input_dict)

    return {f'{split}accuracy': acc, f'{split}rocauc': rocauc_dict['rocauc']}


def test(args, model=None, prefix=None, checkpoints=None, val=True):
    rundir = Path('runs', args.run, 'checkpoints')
    if prefix is not None:
        rundir = Path(prefix, rundir)

    if not rundir.exists():
        raise FileNotFoundError('Run folder not found')

    if checkpoints is None:
        checkpoints = [file_ for file_ in os.listdir(
            rundir) if '.pth' in file_]

    if len(checkpoints) == 0:
        raise FileNotFoundError('No checkpoint found')

    device = torch.device(
        'cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')

    # Download and process data at './dataset/ogbg_molhiv/'
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root='dataset/')

    split_idx = dataset.get_idx_split()
    valid_loader = DataLoader(
        dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(
        dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False)

    evaluator = Evaluator(name="ogbg-molhiv")

    results = {'model': [], 'test_rocauc': [], 'val_rocauc': []}

    with tqdm(checkpoints) as tcheckpoints:
        for i, chk in enumerate(tcheckpoints):
            tcheckpoints.set_description(
                f'Evaluating {chk} {i+1} / {len(tcheckpoints)}')

            if model is None:
                model = models.SAGE(dataset.num_features, args.emb_dim,
                                    args.layers, aggrtype='attn', readout='attn').to(device)

            state_dict = torch.load(Path(rundir, chk))
            model.load_state_dict(state_dict)

            metrics_test = eval(model, test_loader, evaluator,
                                'test', device, progress=False)
            if val:
                metrics_val = eval(model, valid_loader, evaluator,
                                   'val', device, progress=False)

            # append results
            results['model'].append(chk)
            results['test_rocauc'].append(metrics_test['test_rocauc'])
            results['val_rocauc'].append(metrics_val['val_rocauc'] if val else '-')

    results_numerical = deepcopy(results)

    # highlight best models
    max_idx = results['test_rocauc'].index(max(results['test_rocauc']))
    results['model'][max_idx] = f"**{results['model'][max_idx]}**"
    results['test_rocauc'][max_idx] = f"**{results['test_rocauc'][max_idx]}**"

    if val:
        max_idx = results['val_rocauc'].index(max(results['val_rocauc']))
        results['model'][max_idx] = f"**{results['model'][max_idx]}**"
        results['val_rocauc'][max_idx] = f"**{results['val_rocauc'][max_idx]}**"

    # to markdown
    str_res = DataFrame.from_dict(results).to_markdown(index=False)

    # save results table
    if not args.nosave:
        savepath = Path('runs', args.run, 'results.md')
        if prefix is not None:
            savepath = Path(prefix, savepath)
        print(f'Saving to {savepath}')
        with open(savepath, 'w') as f:
            print(f'\nResults for: {args.run}\n')
            print(str_res)
            f.write(f'Results for: {args.run}\n\n')
            f.write(str_res)

    return results_numerical


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
    parser.add_argument('--nosave', action=argparse.BooleanOptionalAction,
                        default=False, help='results will not be saved on file')
    parser.add_argument('--run', type=str, help='run name')

    args = parser.parse_args()

    _ = test(args)
