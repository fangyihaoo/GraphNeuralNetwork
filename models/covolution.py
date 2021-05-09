import torch
import torch.nn as nn
from .basic_module import BasicModule


class CovNet(BasicModule):
    
    def __init__(self,
        num_feature: int = 16,            # number of features
        num_class: int = 7,               # number of classes
        num_cov: int = 5,                # number of convolution layer
        p: float = 0.2,                  # dropout rate
        lam: float = 1e-4                # lambda of regularization         
        ):
        super(CovNet, self).__init__()        
        self.cov_list = [GCNConv(num_feature,  num_feature, cached = True, normalize = TRUE) for i in range(num_cov)]
        self.dropout = [nn.Dropout(p = p) for i in range(num_cov)]
        self.block = nn.Sequential(*[item for pair in zip(self.cov_list, self.dropout) for item in pair])
        self.output = GCNConv(num_feature,  num_class, cached = True, normalize = TRUE)
        
    def forward(self):
        