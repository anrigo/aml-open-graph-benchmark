# %
import argparse
import os
from pathlib import Path
import torch
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import models
import wandb

from utils import accuracy


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


def train(args):
    '''Training loop'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Download and process data at './dataset/ogbg_molhiv/'
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root='dataset/')

    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(
        dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(
        dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False)
    # test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)

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

    model = models.SimpleGCN(dataset.num_features, args.emb_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = torch.nn.BCELoss().to(device)

    evaluator = Evaluator(name="ogbg-molhiv")

    val_rocauc = torch.zeros(args.epochs+1)

    # training loop
    print('Training...')
    for epoch in range(1, args.epochs + 1):
        with tqdm(train_loader, unit='batch') as tepoch:
            tepoch.set_description(f'Epoch {epoch} / {args.epochs}')

            for _, batch in enumerate(tepoch):
                batch.to(device)
                model.train()
                optimizer.zero_grad()

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

        outdir = Path('runs', args.run)
        if not outdir.exists():
            os.makedirs(outdir)

        torch.save(model.state_dict(), Path(outdir, f'{epoch}.pth'))
    
    print(f'Best val_rocauc: {val_rocauc.max()} at epoch {val_rocauc.argmax()}')


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='total epochs')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='batch size')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--dw', action=argparse.BooleanOptionalAction,
                        default=False, help='disable wandb, defaults to False')
    parser.add_argument('--run', type=str, help='run name')

    args = parser.parse_args()

    # args.run = 'simple-gcn'
    # args.epochs = 1
    # args.batch_size = 6
    # args.dw = True

    train(args)
