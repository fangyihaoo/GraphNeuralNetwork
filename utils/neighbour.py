import torch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import torch_geometric.transforms as T



def NewNegihbour(dat, C = 3):
    '''
    Find the nearest neighbour according to the adjacency matrix

    Input: Pytorch geometric data object,  C is constant. Default 3

    Output: Edge index corresponding to the new adjacency matrix

    '''
    flag = False
    if dat['edge_index']:
        adj = to_dense_adj(dat['edge_index'])
    elif dat['adj_t']:
        adj = data['adj_t'].to_dense()
        flag = True
    else:
        raise KeyError("Sorry, there is no edge information")
    
    n = torch.tensor(adj.size()[0])
    w = torch.sqrt(torch.log(n))
    adj2 = adj @ adj.T*(1./n)
    D = torch.cdist(adj2, adj2, p = float('inf'))
    qt = torch.quantile(D, torch.log(n)/(w*torch.sqrt(n))*C, dim=1, keepdim=True)
    newadj = (D < qt)*1
    newadj = (newadj + newadj.T)/2

    
    if flag:
        Trans = T.ToSparseTensor()
        dat['edge_index'] = dense_to_sparse(newadj)[0]
        Trans(dat)
    else:
        dat['edge_index'] = dense_to_sparse(newadj)[0]

    return dat

