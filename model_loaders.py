from torch import nn
from my_own_net import MyOwnNet
import torch


def default_loader(ckpt_path='./models/mnist_fc_net.pth') -> nn.Module:
    model = MyOwnNet()
    state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model
