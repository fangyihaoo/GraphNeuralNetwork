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

    max_epoch = 1000 # number of epoch


    #===============================================
    '''
    Preconditioner
    '''
    eps = 0.001

    update_freq = 50
    
    alpha = None

    gamma = None

    precond = False

    #===============================================

    lr = 1e-3 # initial learning rate

    momentum = 0.9

    layer = 2

    layerwise = False

    resampling = False

    dropedge = None

    dropout = 0.0 # dropout rate
    
    lamb = None # lambda for regulation 

    ite = 10  # number of iteration


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