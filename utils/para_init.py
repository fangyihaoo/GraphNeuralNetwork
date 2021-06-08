import torch
import torch.nn as nn
import torch.nn.init as init


@torch.no_grad()
def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    gain = init.calculate_gain('relu')
    # if isinstance(m, nn.Linear):
    #     init.xavier_normal_(m.weight.data, gain=gain)
    #     if m.bias is not None:
    #         init.zeros_(m.bias)

    # if isinstance(m, nn.Linear):
    #     init.xavier_uniform_(m.weight.data, gain=gain)
    #     if m.bias is not None:
    #         init.zeros_(m.bias)


    
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)

    # if isinstance(m, nn.Linear):
    #     init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
    #     if m.bias is not None:
    #         init.zeros_(m.bias)


if __name__ == '__main__':
    pass