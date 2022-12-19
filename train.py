import argparse
import os
from pathlib import Path
import torch
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import models
import wandb
from evaluate import eval


def grid_search(args):
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root='dataset/')

    learning_rates = [0.001]
    batch_sizes = [64]
    num_layers = [6, 8, 10]
    hidden_dims = [300]

    reduction = {
        6: 0.25,
        8: 0.40,
        10: 0.50
    }

    for lr in learning_rates:
        for bs in batch_sizes:
            for layers in num_layers:
                for h in hidden_dims:
                    model = models.DiffPool(
                        dataset.num_features, h, layers, reduce_to=0.75, aggrtype='max', aggrpool='max', readout='attn')
                    args.lr = lr
                    args.batch_size = bs


def train(args, model=None, prefix=None):
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
        train_metrics = eval(model, train_loader, evaluator, 'train', device)
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
    parser.add_argument('--lr', type=float, default=0.001, help='batch size')
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

    args = parser.parse_args()

    if args.tune:
        args.dry
        args.dw = True
        args.nosave = True

        grid_search(args)
    else:
        if args.dry:
            args.dw = True
            args.nosave = True

        _ = train(args)
