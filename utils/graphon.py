import torch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import torch_geometric.transforms as T



def NewNegihbour(dat) -> None:
    '''
    Find the nearest neighbour according to the adjacency matrix (no edge information)

    Input: Pytorch geometric data object,  C is constant. Default 3

    '''
    flag = False
    if 'edge_index' in dat.keys:
        adj = to_dense_adj(dat['edge_index'])[0]
    elif 'adj_t' in dat.keys:
        adj = dat['adj_t'].to_dense()
        flag = True
    else:
        raise KeyError("Sorry, there is no edge information")
    
    n = torch.tensor(adj.size()[0])
    adj2 = adj @ adj.T*(1./n)
    D = torch.cdist(adj2, adj2, p = float('inf'))
    qt = torch.quantile(D, torch.sqrt(torch.log(n)/n), dim=1, keepdim=True)
    newadj = (D < qt)*1
    newadj = (newadj + newadj.T)/2
    
    if flag:
        Trans = T.ToSparseTensor()
        dat['edge_index'] = dense_to_sparse(newadj)[0]
        Trans(dat)
    else:
        dat['edge_index'] = dense_to_sparse(newadj)[0]

    return None



def NeighbourSmoothing(dat): 

    '''
    Python implementation of the algorithm proposed by Zhang, Y., Levina, E. and Zhu, J. (2016) 
    'Estimating neighborhood edge probabilities by neighborhood smoothing.' arXiv: 1509.08588. 
    
    Input:
        Data object from the pytorch geometric containg 'edge_index' or 'adj_t'
    
    Output: 
        Estimated probaility matrix 
    '''

    if 'edge_index' in dat.keys:
        adj = to_dense_adj(dat['edge_index'])[0]
    elif 'adj_t' in dat.keys:
        adj = dat['adj_t'].to_dense()
    else:
        raise KeyError("Sorry, there is no edge information")

    n = torch.tensor(adj.size()[0])
    adj2 = adj @ adj.T*(1./n)
    D = torch.cdist(adj2, adj2, p = float('inf'))
    K = D < torch.quantile(D, torch.sqrt(torch.log(n)/n), dim=1, keepdim=True)
    P = adj @ (K * (1 / (torch.sum(K, 0) + 1e-10)))
    
    return  (P + P.T) * 0.5


def AdjacencySampling(P, sym = True):

    '''
    Generating the adjacency matrix according to link probability matrix

    Args:
        P: link probability from NeighbourSmoothing function

    Output：
        Random Sampling of Adjacency matrix 
    '''
    adj = torch.rand(P.size(), device = P.device) < P
    if sym:
        uper = torch.triu(adj)
        adj = uper + uper.T
    edge_index, _ = dense_to_sparse(adj.fill_diagonal_(0))

    return edge_index