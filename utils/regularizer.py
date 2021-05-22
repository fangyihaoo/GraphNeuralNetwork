import torch
from torch import Tensor
 
def regularizer(model, norm = 'row') -> Tensor:

    '''
    The regularization term in the loss function.

    norm:
        'row': L2,1 norm of weight parameters(row sparsity)
        'l2': L2 norm of weight parameters
        'l1': L1 norm of weight parameters
    '''
    # loss = torch.tensor(0., device = next(model.parameters()).device)
    loss = 0

    if norm == 'row':
        for name, param in model.named_parameters():
            if 'weight' in name:
                loss += torch.sum(torch.norm(param, dim=0))
    elif norm == 'l2':
        for name, param in model.named_parameters():
            if 'weight' in name:
                loss += torch.norm(param)
    elif norm == 'l1':
        # l1_penalty = torch.nn.L1Loss(size_average=False)
        for name, param in model.named_parameters():
            if 'weight' in name:
                loss += torch.norm(param, p = 1)     
    else:
        raise ValueError('There is no such option for the required norm')
    
    return loss


