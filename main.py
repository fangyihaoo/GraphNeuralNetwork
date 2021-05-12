import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import models
from config import opt
from utils import seed_setup
from utils import weight_init
from utils import Optim



def train(**kwargs):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Planetoid(root='./data/Planetoid', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]
    data.to(device)
    opt._parse(kwargs)

    model = getattr(models, opt.model)(
        num_feature = dataset.num_features, 
        hidden_channels = dataset.num_features, 
        num_class = dataset.num_classes,
        num_cov = opt.layer,
        p = opt.p
        )

    if opt.load_model_path:
        model.load(opt.load_model_path)

    seed_setup()
    model.to(device)
    model.apply(weight_init)
    criterion = torch.nn.CrossEntropyLoss()
    opti = Optim(model.parameters(), opt)
    optimizer = opti.optimizer
    best_val_acc = 0.
    test_acc = 0.

    for epoch in range(opt.max_epoch + 1):
        model.train()
        optimizer.zero_grad()  
        out = model(data.x, data.edge_index)  
        loss = criterion(out[data.train_mask], data.y[data.train_mask]) 

        if opt.regu:
            regularizer = torch.tensor(0.)

            for name, param in model.named_parameters():
                if 'weight' in name:
                    regularizer = regularizer + torch.sum(torch.norm(param, dim=0))
            
            loss = loss + regularizer

        loss.backward()
        optimizer.step()

        train_acc, val_acc, tmp_test_acc = test(model, data)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        
        log = 'Epoch: {:04d}, Train: {:.5f}, Val: {:.5f}, Test: {:.5f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))


@torch.no_grad()
def test(model, data):
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

if __name__=='__main__':
    import fire
    fire.Fire()

