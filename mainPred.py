import torch
from data import MyPlanetoid
from torch_geometric.transforms import NormalizeFeatures
import torchnet.meter as meter
import models
from config import opt
from utils import seed_setup
from utils import weight_init
from utils import Optim
from utils import regularizer
from utils import KFAC

def train(**kwargs):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt._parse(kwargs)
    dataset = MyPlanetoid(name=opt.data, split=opt.split, transform=NormalizeFeatures())
    data = dataset[0]
    data.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    acc_meter = meter.AverageValueMeter()

    for i in range(opt.ite + 1):

        model = getattr(models, opt.model)(
        num_feature = dataset.num_features, 
        hidden_channels = opt.num_hidden, 
        num_class = dataset.num_classes,
        num_cov = opt.layer,
        p = opt.rate
        )
        
        if opt.load_model_path:
            model.load(opt.load_model_path)

        opti = Optim(model.parameters(), opt)
        optimizer = opti.optimizer

        model.to(device)
        model.apply(weight_init)
        best_val_acc = 0.
        test_acc = 0.

        for epoch in range(opt.max_epoch + 1):
            model.train()
            optimizer.zero_grad()  
            out = model(data.x, data.edge_index)  
            loss = criterion(out[data.train_mask], data.y[data.train_mask]) 

            if opt.lamb:
                loss +=  opt.lamb*regularizer(model, opt.norm)

            loss.backward()
            
            ## preconditioner
            # if opt.precond:
            #     preconditioner = KFAC(
            #             model, 
            #             opt.eps, 
            #             sua=False, 
            #             pi=False, 
            #             update_freq = opt.update_freq,
            #             alpha = opt.alpha if opt.alpha is not None else 1.,
            #             constraint_norm = False
            #         )
            #     lam = (float(epoch)/float(epochs))**opt.gamma if opt.gamma is not None else 0.
            #     label = label = out.max(1)[1]
            #     label.requires_grad = False
            #     loss += lam*criterion(out[~data.train_mask], label[~data.train_mask]) 
            #     preconditioner.step(lam=lam)

            optimizer.step()
            train_acc, val_acc, tmp_test_acc = test(model, data)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc

        acc_meter.add(test_acc)

    log = 'Mean: {:.5f}, Std: {:.3f}'
    print(log.format(acc_meter.value()[0], acc_meter.value()[1]))


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

