import torch
from tqdm import tqdm
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
