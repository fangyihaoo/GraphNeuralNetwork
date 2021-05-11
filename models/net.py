import torch
import torch.nn as nn
from .basic_module import BasicModule
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch import Tensor


class CovNet(BasicModule):
    '''
    Graph Convolution Network
    '''
    def __init__(self,
        num_feature: int = 16,            # number of features
        hidden_channels: int = 16,        # number of hidden channel
        num_class: int = 7,               # number of classes
        num_cov: int = 5,                 # number of convolution layer
        act = nn.ReLU(),                  # activation function
        p: float = None,                  # dropout rate     
        ):
        super(CovNet, self).__init__() 

        self.num_cov = num_cov
        self.p = p
        self.act = act

        for i in range(self.num_cov):
            setattr(self,f'conv{i}', GCNConv(num_feature, hidden_channels, cached=True))        # because its the node classification, so cached = True

        if self.p:
            self.drop = nn.Dropout(p = self.p)

        self.output = GCNConv(num_feature, num_class, cached=True)

    def forward(self, x: Tensor, edge_index) -> Tensor:     
        for i in range(self.self.num_cov):
            x = getattr(self, f'conv{i}')(x, edge_index)
            x = self.act(x)
            if self.p:
                x = self.drop(x)
        
        x = self.output(x, edge_index)
        return x


# class SageNet(BasicModule):
#     '''
#     GraphSAGE Neural Network
#     '''

#     def __init__(self,
#         num_feature: int = 16,                # number of features
#         hidden_channels: int = 16,            # number of hidden channel
#         num_class: int = 7,                   # number of classes
#         num_sage: int = 5,                    # number of sage layer
#         act = nn.ReLU(),                      # activation function
#         p: float = None,                      # dropout rate
#         ):
        
#         self.num_sage = num_sage
#         self.p = p
#         self.act = act

#         for i in range(self.num_sage):
#             setattr(self,f'sage{i}', SAGEConv(num_feature, hidden_channels))        

#         self.output = SAGEConv(num_feature, num_class)

#     def forward(self, )





class AttNet(BasicModule):
    


