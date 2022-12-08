# %
import argparse
import torch
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from models import GCN
import wandb


@torch.no_grad()
def eval(model, loader, evaluator):
    '''Evaluate the model'''
    device = model.device
    model.eval()
    y_true = []
    y_pred = []

    for _, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        pred = model(batch)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


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

    model = GCN(dataset.num_features, args.emb_dim,
                dataset.num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = torch.nn.BCELoss()

    evaluator = Evaluator(name="ogbg-molhiv")

    # training loop
    print('Training...')
    for epoch in range(1, args.epochs + 1):
        with tqdm(train_loader, unit='batch') as tepoch:
            tepoch.set_description(f'Epoch {epoch} / {args.epochs}')

            for _, batch in enumerate(tepoch):
                batch.to(device)
                model.train()
                optimizer.zero_grad()

                print(batch)
                out = model(batch.x, batch.edge_index, batch.edge_attr)
                print(out.shape)

                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()

                # logging
                tepoch.set_postfix(loss=loss.item())

                wandb.log({
                    'epoch': epoch,
                    'loss': loss.item()
                })

        # validate
        print('Validating..')
        train_metrics = eval(model, train_loader, evaluator)
        val_metrics = eval(model, valid_loader, evaluator)
        print(train_metrics)
        print(val_metrics)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='total epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='batch size')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--dw', action=argparse.BooleanOptionalAction,
                        default=False, help='disable wandb, defaults to False')
    parser.add_argument('--run', type=str, help='run name')

    args = parser.parse_args()

    args.run = 'simple-gcn'
    args.dw = True

    train(args)
