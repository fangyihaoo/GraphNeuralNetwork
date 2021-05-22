import torch
import torch.nn as nn
from .basic_module import BasicModule
from utils import AdjacencySampling
from torch_geometric.nn import GCNConv
from torch import Tensor



class CovNet(BasicModule):
    '''
    Graph Convolution Network
    '''
    def __init__(self,
        num_feature: int = 16,            # number of features
        hidden_channels: int = 64,        # number of hidden channel
        num_class: int = 7,               # number of classes
        num_cov: int = 5,                 # number of convolution layer, at least 2
        act = nn.ReLU(),                  # activation function
        dp: float = None,                  # dropout rate     
    ):
        super(CovNet, self).__init__() 

        self.num_cov = num_cov
        self.dp = dp
        self.act = act

        self.conv0 = GCNConv(num_feature, hidden_channels, cached=True)

        for i in range(1, self.num_cov - 1):
            setattr(self,f'conv{i}', GCNConv(hidden_channels, hidden_channels, cached=True))

        if self.dp:
            self.drop = nn.Dropout(p = self.dp)

        self.output = GCNConv(hidden_channels, num_class, cached=True)

    def forward(self, x: Tensor, edge_index) -> Tensor:  
        
        for i in range(self.num_cov - 1):
            x = getattr(self, f'conv{i}')(x, edge_index)
            x = self.act(x)
            if self.dp:
                x = self.drop(x)
        
        x = self.output(x, edge_index)
        return x


class ResamplingNet(BasicModule):
    '''
    Resampling Graph Convolution Net
    
    Each layer, the Laplacian matrix is resampling according to the graphon estimation
    '''
    def __init__(self,
        prob,                             # link probability matrix, cached
        num_feature: int = 16,            # number of features
        hidden_channels: int = 64,        # number of hidden channel
        num_class: int = 7,               # number of classes
        num_cov: int = 5,                 # number of convolution layer, at least 2
        act = nn.ReLU(),                  # activation function
        dp: float = None,                  # dropout rate
    ):
        super(ResamplingNet, self).__init__() 

        self.num_cov = num_cov
        self.dp = dp
        self.act = act

        self.conv0 = GCNConv(num_feature, hidden_channels, cached=True)

        for i in range(1, self.num_cov - 1):
            setattr(self,f'conv{i}', GCNConv(hidden_channels, hidden_channels, cached=True))

        if self.dp:
            self.drop = nn.Dropout(p = self.dp)

        self.output = GCNConv(hidden_channels, num_class, cached=True)

        self.prob = prob

    def forward(self, x: Tensor, edge_index) -> Tensor:  
        
        for i in range(self.num_cov - 1):
            x = getattr(self, f'conv{i}')(x, edge_index)
            x = self.act(x)
            if self.dp:
                x = self.drop(x)
            edge_index = AdjacencySampling(self.prob)
        
        x = self.output(x, edge_index)
        return x