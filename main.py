import torch


















def train(**kwarg):













    regularizer = torch.tensor(0.)

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(torch.sum(torch.pow(param, 2),0))