import torch
import torch.nn as nn
import os.path as osp
from torch_geometric.nn import GCNConv, JumpingKnowledge, GINConv, GATConv
from .basic_module import BasicModule
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from utils import AdjacencySampling
from torch import Tensor
from typing import Callable


class CovNet(BasicModule):

    '''
    Graph Convolution Network
    '''
    def __init__(self,
        num_feature: int = 16,                                   # number of features
        hidden_channels: int = 64,                               # number of hidden channel
        num_class: int = 7,                                      # number of classes
        num_cov: int = 5,                                        # number of convolution layer, at least 2
        act: Callable[..., Tensor] = nn.ReLU(),                  # activation function
        dropout: float = 0.0,                  # dropout rate     
    ) -> None:
        super(CovNet, self).__init__() 

        self.num_cov = num_cov
        self.drop = nn.Dropout(p = dropout)
        self.act = act

        self.conv0 = GCNConv(num_feature, hidden_channels, cached=True)

        for i in range(1, self.num_cov - 1):
            setattr(self,f'conv{i}', GCNConv(hidden_channels, hidden_channels, cached=True))

        self.output = GCNConv(hidden_channels, num_class, cached=True)

    def forward(self, x: Tensor, edge_index: Tensor, link_prob: Tensor = None) -> Tensor:  
        
        for i in range(self.num_cov - 1):
            x = getattr(self, f'conv{i}')(x, edge_index)
            x = self.act(x)
            x = self.drop(x)
            if link_prob:
                if self.training:
                    edge_index = AdjacencySampling(link_prob)
        x = self.output(x, edge_index)
        return x

    def __repr__(self):
        return self.__class__.__name__



class GIN0WithJK(BasicModule):
    '''
    Implimentation of 'Representation Learning on Graphs with Jumping Knowledge Networks'

    Node Classification Modified Version from https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py#L54-L106
    mode (string): The aggregation scheme to use
            (:obj:`"cat"`, :obj:`"max"` or :obj:`"lstm"`).
    '''

    def __init__(self, 
        num_feature: int = 16,                                        # number of features
        hidden_channels: int = 64,                                    # number of hidden channel
        num_class: int = 7,                                           # number of classes
        num_cov: int = 5,                                             # number of convolution layer, at least 2
        act: Callable[..., Tensor] = nn.ReLU(),                       # activation function
        dropout: float = 0.0,  
        mode='max'):

        super(GIN0WithJK, self).__init__()
        self.act = act
        self.drop = nn.Dropout(p = dropout)

        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(num_feature, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, hidden_channels),
                self.act,
                nn.BatchNorm1d(hidden_channels),
            ), train_eps=False)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_cov - 1):
            self.convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_channels, hidden_channels),
                        self.act,
                        nn.Linear(hidden_channels, hidden_channels),
                        self.act,
                        nn.BatchNorm1d(hidden_channels),
                    ), train_eps=False))
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = nn.Linear(num_cov * hidden_channels, hidden_channels)
        else:
            self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, num_class)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self,  x: Tensor, edge_index: Tensor, link_prob: Tensor = None) -> Tensor:
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            if link_prob:
                if self.training:
                    edge_index = AdjacencySampling(link_prob)
            x = conv(x, edge_index)
            xs += [x]
        x = self.jump(xs)
        x = self.act(self.lin1(x))
        x = self.drop(x)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__



class GATNet(BasicModule):
    '''
    Graph Attention Network
    '''

    def __init__(self,
        num_feature: int = 16,                                        # number of features
        hidden_channels: int = 64,                                    # number of hidden channel
        num_class: int = 7,                                           # number of classes
        num_cov: int = 5,                                             # number of convolution layer, at least 2
        act: Callable[..., Tensor] = nn.ReLU(),                       # activation function
        dropout: float = 0.0,                                         # dropout rate
    ):
        super(GATNet, self).__init__()
        self.num_cov = num_cov
        self.act = act
        self.drop = nn.Dropout(p = dropout)

        self.conv0 = GATConv(num_feature, hidden_channels, heads = 1, dropout = dropout)         # input layer, heads set to 1 to avoid dimension explosion in multilayer setting.

        # hidden layers
        for i in range(1, self.num_cov - 1):
            setattr(self,f'conv{i}', GATConv(hidden_channels, hidden_channels, heads = 1, dropout = dropout))

        # output layer
        self.output = GATConv(hidden_channels, num_class, heads = 1, dropout = dropout)     

    def forward(self,  x: Tensor, edge_index: Tensor, link_prob: Tensor = None) -> Tensor:

        for i in range(self.num_cov - 1):
            x = getattr(self, f'conv{i}')(x, edge_index)
            x = self.act(x)
            x = self.drop(x)
            if link_prob:
                if self.training:
                    edge_index = AdjacencySampling(link_prob)
        x = self.output(x, edge_index)
        return x

    def __repr__(self):
        return self.__class__.__name__

