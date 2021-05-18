import os.path as osp
import torch
from torch_geometric.datasets import Planetoid



def MyPlanetoid(name, split="public", num_train_per_class=20,
                num_val=500, num_test=1000, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)

    '''
    Data encapsulation for different split scheme

    split(string)
        complete:
            "CayleyNets: Graph Convolutional Neural Networks with Complex Rational Spectral Filters"
            1708 training samples, 500 validation samples, 500 test samples
        public:
            "Revisiting Semi-Supervised Learning with Graph Embeddings"
            140 training samples, 500 validation samples, 1000 test samples
        full:
            "FASTGCN: FAST LEARNING WITH GRAPH CONVOLUTIONAL NETWORKS VIA IMPORTANCE SAMPLING"
            1208 training samples, 500 validation samples, 1000 test samples
        random:
            check online "https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/planetoid.html"
    '''

    if split == 'complete':
        dataset = Planetoid(path, name, transform = transform)
        dataset[0].train_mask.fill_(False)
        dataset[0].train_mask[:dataset[0].num_nodes - 1000] = 1
        dataset[0].val_mask.fill_(False)
        dataset[0].val_mask[dataset[0].num_nodes - 1000:dataset[0].num_nodes - 500] = 1
        dataset[0].test_mask.fill_(False)
        dataset[0].test_mask[dataset[0].num_nodes - 500:] = 1
    else:
        dataset = Planetoid(path, name, split = split, num_train_per_class = num_train_per_class, num_val=num_val, num_test = num_test, transform = transform)

    return dataset