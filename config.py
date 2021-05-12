import warnings

class DefaultConfig(object):
    '''
    default setting, can be changed via command line
    '''

    model = 'CovNet'

    # test_data_root = './data/exact_sol'

    load_model_path = None
    #'checkpoints/model.pth' # path for trained model
    
    method = 'adam'

    max_epoch = 2000 # number of epoch

    lr = 0.001 # initial learning rate

    layer = 5

    regu = False
    
    rate = None # dropout rate


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
                print(k,getattr(self,k))


opt = DefaultConfig()