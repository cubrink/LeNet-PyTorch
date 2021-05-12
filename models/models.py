import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=1, padding=0, **kwargs):
        super(ConvBlock, self).__init__()
        self.C = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
        self.A = nn.Sigmoid()
        self.P = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.C(x)   # Convolution
        x = self.A(x)   # Activation
        x = self.P(x)   # Pooling
        return x

class FeedForward(nn.Module):
    def __init__(self, in_features, out_features, act=nn.Sigmoid, **kwargs):
        super(FeedForward, self).__init__()
        self.D = nn.Linear(in_features=in_features, out_features=out_features)
        self.A = act()

    def forward(self, x):
        x = self.D(x)   # Dense layer
        x = self.A(x)   # Activation
        return x

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.C1 = ConvBlock(in_channels=1, out_channels=6, kernel_size=(5,5), padding=2, stride=1)
        self.C2 = ConvBlock(in_channels=6, out_channels=16, kernel_size=(5,5), padding=0, stride=1)
        self.F = nn.Flatten()
        self.D3 = FeedForward(in_features=5*5*16, out_features=120)
        self.D4 = FeedForward(in_features=120, out_features=84)
        self.D5 = FeedForward(in_features=84, out_features=10, act=nn.Identity)

    def forward(self, x):
        x = self.C1(x)  # Conv
        x = self.C2(x)  # Conv
        x = self.F(x)   # Flatten
        x = self.D3(x)  # Dense
        x = self.D4(x)  # Dense
        x = self.D5(x)  # Dense
        return x

