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
            return optim.Adagrad(self.params, lr = self.lr)

        elif self.method == 'rmsprop':
            return optim.RMSProp(self.params, lr = self.lr, alpha = 0.9)

        elif self.method == 'adam':
            return optim.Adam(self.params, lr=self.lr)
        
        # to use SGD, we need to modify the train part and dataset
        # elif self.method == 'sgd':
        #     return optim.SGD(self.params, lr = self.lr)

        else:
            raise RuntimeError("Invalid optim method: " + self.method)