import torch
import torch.nn as nn

class LeNetBlock(nn.Module):
    """
    Conv block used in LeNet

    CNN -> sigmoid -> Avg pooling
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super(LeNetBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs) # kwargs
        self.act = nn.Sigmoid()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class FullyConnected(nn.Module):
    """
    Fully connected layer -> activation function
    """
    def __init__(self, in_features, out_features, act=None, **kwargs):
        super(FullyConnected, self).__init__()
        self.F = nn.Linear(in_features=in_features, out_features=out_features)
        if act is None:
            act = nn.Identity()
        self.act = act(**kwargs)
    
    def forward(self, x):
        x = self.F(x)
        x = self.act(x)
        return x

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.C1 = LeNetBlock(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.C2 = LeNetBlock(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.F3 = FullyConnected(in_features=400, out_features=120, act=nn.Sigmoid)
        self.F4 = FullyConnected(in_features=120, out_features=84, act=nn.Sigmoid)
        self.F5 = FullyConnected(in_features=84, out_features=10, act=nn.LogSoftmax, **dict(dim=1))
    
    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        x = nn.Flatten()(x)
        x = self.F3(x)
        x = self.F4(x)
        x = self.F5(x)
        return x

        
