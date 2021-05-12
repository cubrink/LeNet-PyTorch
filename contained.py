import torch
import torch.nn as nn
import torchvision.datasets
import torch.nn.functional as F
import torchvision.transforms as transforms

assert torch.cuda.is_available()
device = 'cuda:0'

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

preprocess = transforms.Compose([transforms.ToTensor()])

# Get data
train_data = torchvision.datasets.MNIST('./mnist/', train=True, download=True, transform=preprocess)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=False)

# Define model (~LeNet)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.linear1 = nn.Linear(in_features=5*5*16, out_features=120)
        self.linear2 = nn.Linear(in_features=120, out_features=84)
        self.linear3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


model = ConvNet().to(device)
model.train()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss().to(device)

total_loss = 0
for idx, (X, y) in enumerate(train_loader):
    print(f"Batch {idx}", end="\r")
    X, y = X.to(device), y.to(device)
    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y_pred, y)
    total_loss += loss.item()
    loss.backward()
    optimizer.step()
print("Total loss =", total_loss)