import warnings

class DefaultConfig(object):
    '''
    default setting, can be changed via command line
    '''

    model = 'CovNet'

    data = 'Cora'

    load_model_path = None
    #'checkpoints/model.pth' # path for trained model
    
    method = 'adam'

    split = 'public'

    norm = 'row'

    num_hidden = 64

    max_epoch = 300 # number of epoch

    lr = 0.0005 # initial learning rate

    layer = 2

    rate = None # dropout rate
    
    lamb = None # lambda for regulation 

    ite = 100  # number of iteration


    def _parse(self, kwargs):
        '''
        update parameters according to user preference
        '''
        for k,v in kwargs.items():
            if not hasattr(self,k):
                warnings.warn("Warning: opt has not attribut %s" %k)
            setattr(self,k,v)

        print('user config:')
        for k,v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                if '_parse' in k:
                    continue
                else:
                    print(k,getattr(self,k))


opt = DefaultConfig()