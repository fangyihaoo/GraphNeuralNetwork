import torch
import os.path as osp
from torch_geometric.transforms import NormalizeFeatures
import torchnet.meter as meter
import models
from data import MyPlanetoid
from config import opt
from utils import weight_init
from utils import Optim
from utils import regularizer
from utils import KFAC
from utils import write_excel
from utils import AdjacencySampling

def train(**kwargs):

    # setup
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'result', "")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt._parse(kwargs)
    dataset = MyPlanetoid(name=opt.data, split=opt.split, transform=NormalizeFeatures())
    data = dataset[0]
    data = data.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    acc_meter = meter.AverageValueMeter()
    key = {'num_feature' :dataset.num_features, 
        'hidden_channels':opt.num_hidden,
        'num_class':dataset.num_classes,
        'num_cov':opt.layer,
        'dp':opt.rate}
    if opt.model == 'ResamplingNet':
        prop = torch.load(path + opt.data + 'prop.pt', map_location = device)

    for _ in range(opt.ite + 1):

        model = getattr(models, opt.model)(**key)
        if opt.load_model_path:
            model.load(opt.load_model_path)

        opti = Optim(model.parameters(), opt)
        optimizer = opti.optimizer

        model.to(device)
        model.apply(weight_init)
        best_val_acc = 0.
        test_acc = 0.
        edge = data['edge_index']

        for _ in range(opt.max_epoch + 1):
            model.train()
            optimizer.zero_grad()

            if opt.model == 'ResamplingNet':
                out = model(data.x, edge, prop)
            else:
                out = model(data.x, edge)

            loss = criterion(out[data.train_mask], data.y[data.train_mask]) 
            if opt.lamb:
                loss +=  opt.lamb*regularizer(model, opt.norm)

            loss.backward()
            optimizer.step()

            if opt.model == 'ResamplingNet':
                edge = AdjacencySampling(prop)
                _, val_acc, tmp_test_acc = test(model, data, data['edge_index'], prop)
            else:
                _, val_acc, tmp_test_acc = test(model, data, data['edge_index'])

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc

        acc_meter.add(test_acc)
    write_excel(acc_meter.value(), opt, path)


@torch.no_grad()
def test(model, data, edge, prop = None):
    model.eval()
    if model.__class__.__name__ == 'CovNet':
        out = model(data.x, edge)
    else:
        out = model(data.x, edge, prop)
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

