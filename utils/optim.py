import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch import Tensor
from typing import List

class Optim(object):
    '''
    Make optimizer according to config.py
    '''

    def __init__(self, params, config):
        self.params = params  
        self.method = config.method
        self.lr = config.lr
        self._makeOptimizer()

    def _makeOptimizer(self):
        if self.method == 'adagrad':
            self.optimizer =  optim.Adagrad(self.params, lr = self.lr)

        elif self.method == 'rmsprop':
            self.optimizer = optim.RMSProp(self.params, lr = self.lr, alpha = 0.9)

        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr)
        
        elif self.method == 'sgd':
            self.optimizer =  optim.SGD(self.params, lr = self.lr, momentum = config.momentum)

        elif self.method == 'pgnn':
            self.optimizer = Pgnn(self.params, lr = self.lr)

        else:
            raise RuntimeError("Invalid optim method: " + self.method)




class Pgnn(Optimizer):
    r""" Implements Proximal descent on Clipped L1 norm

    It is proposed in 'Nonconvex sparse regularization for deep neural networks and its optimality'

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        tau (float, optional): controls the scale of the parameters (default: 1e-3)
        lambda (float, optional): controls the importance of the regularization term (default: 1e-5)

    """

    def __init__(self, params, lr = 1e-3, tau = 1e-4, lamb = 1e-5) -> None:
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= tau:
            raise ValueError("Invalid learning rate: {}".format(tau))
        if not 0.0 <= lamb:
            raise ValueError("Invalid learning rate: {}".format(lamb))
        defaults = dict(lr = lr, tau = tau, lamb = lamb)
        super(Pgnn, self).__init__(params, defaults)


    @torch.no_grad()
    def step(self, closure=None):
        r"""Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            tau = group['tau']
            lamb = group['lamb']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    self.state[p]
            update(params_with_grad,
                    grads,
                    tau = tau,
                    lamb = lamb,
                    lr = lr)

        return loss


def update(params: List[Tensor],
            grads: List[Tensor],
            *,
            tau: float,
            lamb: float,
            lr: float) -> None:
    r"""
    Functional API that performs update algorithm in equation (4.3) and (4.5) of
    'Nonconvex sparse regularization for deep neural networks and its optimality'

    """
    for i, param in enumerate(params):
        grad = grads[i]
        h = torch.sign(param.detach())*torch.gt(torch.abs(param.detach()), tau)              
        param.add_(-lr*grad.add(-lamb*h/tau))
        u = param.detach().clone()                                                  
        param.add_(-torch.sign(u)*lr*lamb/tau).mul_(torch.gt(torch.abs(u), lr*lamb/tau))
