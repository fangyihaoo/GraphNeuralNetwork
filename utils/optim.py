import torch.optim as optim

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
            return optim.SGD(self.params, lr = self.lr, momentum = config.momentum)

        else:
            raise RuntimeError("Invalid optim method: " + self.method)