import argparse
import os
from pathlib import Path
import torch
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import models
import wandb
from evaluate import test, eval
from pandas import DataFrame


def ogb_eval(args, k=5, offset=0):
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root='dataset/')
    args.dw = True
    args.epochs = 200
    args.batch_size = 64
    args.layers = 6
    args.emb_dim = 300
    args.lr = 0.001

    config = f'{args.run}, layers: {args.layers}, bs: {args.batch_size}, hidden_dim: {args.emb_dim}, epochs: {args.epochs}, lr: {args.lr}'

    results = {'seed': [], 'test_rocauc': [], 'val_rocauc': []}

    for i in range(k):
        torch.manual_seed(i+offset)
        prefix = 'ogb-eval'
        args.run = str(i+offset)
        print(f'Training with seed: {args.run}')

        model = models.DiffPool(dataset.num_features, args.emb_dim,
                                args.layers, aggrtype='max', aggrpool='max', readout='max')

        best_val, best_ep = train(args, model, False, prefix)

        test_res = test(args, model, prefix, checkpoints=[
                        str(best_ep.item()) + '.pth'], val=False)

        assert len(test_res['test_rocauc']) == 1
        results['seed'].append(i+offset)
        results['test_rocauc'].append(test_res['test_rocauc'][0])
        results['val_rocauc'].append(best_val)

    # compute mean and std over all the seeds
    test_tens = torch.Tensor(results['test_rocauc'])
    val_tens = torch.Tensor(results['val_rocauc'])
    plus_minus = u"\u00B1"
    results['seed'].append('AVG')
    results['test_rocauc'].append(
        f'{test_tens.mean().item()} {plus_minus} {test_tens.std().item()}')
    results['val_rocauc'].append(
        f'{val_tens.mean().item()} {plus_minus} {val_tens.std().item()}')

    # to markdown
    str_res = DataFrame.from_dict(results).to_markdown(index=False)

    # save results table
    partition = str(offset) if offset > 0 else ''
    savepath = Path(prefix, f'{partition}results.md')
    print(f'Saving to {savepath}')
    with open(savepath, 'w') as f:
        print(f'\nResults for: {config}\n')
        print(str_res)
        f.write(f'Results for: {config}\n\n')
        f.write(str_res)


def grid_search(args):
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root='dataset/')

    learning_rates = [0.001]
    batch_sizes = [32, 64]
    num_layers = [6, 8, 10]
    hidden_dims = [300]
    epochs = [200]

    reduce_to = {
        6: 0.25,
        8: 0.40,
        10: 0.50
    }

    args.dry = True
    args.dw = True
    args.nosave = True

    results = {'layers': [], 'bs': [], 'hidden_dim': [],
               'epochs': [], 'lr': [], 'reduce_to': [], 'val_rocauc': []}

    for lr in learning_rates:
        for bs in batch_sizes:
            for layers in num_layers:
                for h in hidden_dims:
                    for e in epochs:
                        config = f'{args.run}, layers: {layers}, bs: {bs}, hidden_dim: {h}, epochs: {e}, lr: {lr}, reduce_to: {reduce_to[layers]}'

                        print('Training ' + config)

                        model = models.DiffPool(
                            dataset.num_features, h, layers, reduce_to=reduce_to[layers], aggrtype='max', aggrpool='max', readout='max')
                        args.lr = lr
                        args.batch_size = bs
                        args.epochs = e

                        best_val, _ = train(args, model, train_eval=False)

                        results['layers'].append(layers)
                        results['bs'].append(bs)
                        results['hidden_dim'].append(h)
                        results['epochs'].append(e)
                        results['lr'].append(lr)
                        results['reduce_to'].append(reduce_to[layers])
                        results['val_rocauc'].append(best_val)

                        print('Results ' + config +
                              f', val_rocauc: {best_val}')

    # highlight the best configuration for the model
    max_idx = results['val_rocauc'].index(max(results['val_rocauc']))

    for k in results.keys():
        results[k][max_idx] = f"**{results[k][max_idx]}**"

    # to markdown
    str_res = DataFrame.from_dict(results).to_markdown(index=False)

    # save results table
    savepath = Path('runs', f'{args.run}-tuning.md')
    print(f'Saving to {savepath}')
    with open(savepath, 'w') as f:
        print(f'\nResults for: {args.run}\n')
        print(str_res)
        f.write(f'Results for: {args.run}\n\n')
        f.write(str_res)


def train(args, model=None, train_eval=True, prefix=None):
    '''Training loop'''
    device = torch.device(
        'cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')

    # Download and process data at './dataset/ogbg_molhiv/'
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root='dataset/')

    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(
        dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(
        dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False)

    # log config
    wandb.init(
        project='aml-ogb',
        config={
            'model/emb_dim': args.emb_dim,
            'dataset/num_classes': dataset.num_classes,
            'dataset/num_features': dataset.num_features
        },
        mode='disabled' if args.dw else None
    )
    wandb.config.update(args)

    # set run name
    wandb.run.name = args.run

    if model is None:
        model = models.DiffPool(dataset.num_features, args.emb_dim,
                                args.layers, aggrtype='max', aggrpool='max', readout='max').to(device)
    else:
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = torch.nn.BCELoss().to(device)

    evaluator = Evaluator(name="ogbg-molhiv")

    val_rocauc = torch.zeros(args.epochs+1)

    # training loop
    print('Training...')
    for epoch in range(1, args.epochs + 1):
        with tqdm(train_loader, unit='batch') as tepoch:
            tepoch.set_description(f'Epoch {epoch} / {args.epochs}')

            for batch in tepoch:
                batch.to(device)
                model.train()
                optimizer.zero_grad()

                if isinstance(model, models.DiffPool):
                    out, link_loss, assign_loss = model(batch.x, batch.edge_index,
                                                        batch.edge_attr, batch.batch)

                    link_loss, assign_loss = link_loss.sum(), assign_loss.sum()

                    loss = criterion(out, batch.y.type(
                        torch.float32)) + link_loss + assign_loss

                    wandb.log({
                        'link_loss': link_loss.item(),
                        'assign_loss': assign_loss.item()
                    }, commit=False)

                else:
                    out = model(batch.x, batch.edge_index,
                                batch.edge_attr, batch.batch)

                    loss = criterion(out, batch.y.type(torch.float32))

                loss.backward()
                optimizer.step()

                # logging
                tepoch.set_postfix(loss='%.3f' % loss.item())

                wandb.log({
                    'epoch': epoch,
                    'loss': loss.item()
                })

        # validate
        print('Validating..')
        train_metrics = eval(model, train_loader, evaluator,
                             'train', device) if train_eval else dict()
        val_metrics = eval(model, valid_loader, evaluator, 'val', device)
        metrics = {
            'epoch': epoch,
            **train_metrics,
            **val_metrics
        }

        wandb.log(metrics)
        val_rocauc[epoch] = metrics['val_rocauc']
        print(metrics)

        if not args.nosave:
            outdir = Path('runs', args.run, 'checkpoints')
            if prefix is not None:
                outdir = Path(prefix, outdir)
            if not outdir.exists():
                os.makedirs(outdir)

            torch.save(model.state_dict(), Path(outdir, f'{epoch}.pth'))

    best_val = val_rocauc.max()
    best_ep = val_rocauc.argmax()
    wandb.log({'best_val_rocauc': best_val})
    print(f'Best val_rocauc: {best_val} at epoch {best_ep}')

    return best_val, best_ep


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='total epochs')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--layers', type=int, default=6,
                        help='number of GNN layers (default: 6)')
    parser.add_argument('--heads', type=int, default=0,
                        help='number of GAT attention heads, if GAT is used (default: 0)')
    parser.add_argument('--dw', action=argparse.BooleanOptionalAction,
                        default=False, help='disable wandb, defaults to False')
    parser.add_argument('--nosave', action=argparse.BooleanOptionalAction,
                        default=False, help='disable checkpoints')
    parser.add_argument('--dry', action=argparse.BooleanOptionalAction,
                        default=False, help='disable checkpoints and wandb logging')
    parser.add_argument('--cpu', action=argparse.BooleanOptionalAction,
                        default=False, help='run on cpu only')
    parser.add_argument('--run', type=str, default='gnn', help='run name')

    parser.add_argument('--tune', action=argparse.BooleanOptionalAction,
                        default=False, help='if True a grid search for hyperparameters tuning will run instead of normal training')
    parser.add_argument('--ogb', action=argparse.BooleanOptionalAction,
                        default=False, help='run OGB evaluation: train and evaluate 10 times with different random seeds')

    args = parser.parse_args()

    if args.tune:
        grid_search(args)
    elif args.ogb:
        ogb_eval(args)
    else:
        if args.dry:
            args.dw = True
            args.nosave = True

        _ = train(args)
