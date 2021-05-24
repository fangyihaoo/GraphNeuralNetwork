import torch

@torch.no_grad()
def eval(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        correct = pred[mask] == data.y[mask]
        acc = float(correct.sum()) / float(mask.sum())
        accs.append(acc)
    model.train()
    return accs


# @torch.no_grad()
# def eval(model, data, edge, prop = None):
#     model.eval()
#     if model.__class__.__name__ == 'CovNet':
#         out = model(data.x, edge)
#     else:
#         out = model(data.x, edge, prop)
#     pred = out.argmax(dim=1)
#     accs = []
#     for _, mask in data('train_mask', 'val_mask', 'test_mask'):
#         correct = pred[mask] == data.y[mask]
#         acc = float(correct.sum()) / float(mask.sum())
#         accs.append(acc)
#     model.train()
#     return accs