import torch
import torch.optim as optim
import os.path as osp
from torch_geometric.transforms import NormalizeFeatures
import torchnet.meter as meter
import models
from data import MyPlanetoid
from config import opt
from utils import weight_init
from utils import Optim
from utils import regularizer
# from utils import KFAC
from utils import write_excel
from utils import AdjacencySampling
from utils import eval
from utils import dropedge

def train(**kwargs):
    #------------------------------------------------------------------------------------------------------------------------------
    # model configuration
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
        'num_lay':opt.layer,
        'dropout':opt.dropout}
    if opt.layerwise or opt.resampling:
        prop = torch.load(path + opt.data + 'prop.pt', map_location = device)
    #------------------------------------------------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------------------------------------------------
    # training part
 
        #---------------------------------------------------------------------------
        # model initialization 
    model = getattr(models, opt.model)(**key)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    opti = Optim(model.parameters(), opt)
    optimizer = opti.optimizer
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 600], gamma=0.5)
    model.to(device)
    model.apply(weight_init)
    val_loss = []
    edge = data['edge_index']
    #---------------------------------------------------------------------------
    for _ in range(opt.max_epoch + 1):
        model.train()
        optimizer.zero_grad()
        if opt.layerwise:
            out = model(data.x, edge, prop)
        else:
            out = model(data.x, edge)
        tmp = criterion(out[data.val_mask], data.y[data.val_mask]) 
        loss = criterion(out[data.train_mask], data.y[data.train_mask]) 
        if opt.lamb:
            loss +=  opt.lamb*regularizer(model, opt.norm)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if opt.layerwise or opt.resampling :
            edge = AdjacencySampling(prop)
        elif opt.dropedge:
            edge = dropedge(edge, opt.dropedge)
        else:
            pass
        val_loss.append(tmp)
    path = path + opt.model + f"{opt.layer}"+f"resampling{opt.resampling}"+f"layerwise{opt.layerwise}" +opt.data+".pt"
    torch.save(val_loss, path)
    #------------------------------------------------------------------------------------------------------------------------------



if __name__=='__main__':
    import fire
    fire.Fire()

