import models
import torch
import models

def setmodel(opt, **kwargs):
    '''
    Setting the neural network model according to config
    '''
    return getattr(models, opt.model)(
        **kwargs
    )










# if opt.model == 'CovNet':
#     model = getattr(models, opt.model)(
#     num_feature = dataset.num_features, 
#     hidden_channels = opt.num_hidden, 
#     num_class = dataset.num_classes,
#     num_cov = opt.layer,
#     p = opt.rate
#     )
# else:
#     model = getattr(models, opt.model)(
#     prob = prop,
#     num_feature = dataset.num_features, 
#     hidden_channels = opt.num_hidden, 
#     num_class = dataset.num_classes,
#     num_cov = opt.layer,
#     p = opt.rate
#     )
