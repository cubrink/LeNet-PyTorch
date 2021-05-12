import torch
import torch.nn as nn
import torchvision.transforms as transforms

from models.models import LeNet
from utils.data import MNIST

assert torch.cuda.is_available()
device = 'cuda:0'

### Hyperparams ###
EPOCHS = 1000
BATCH_SIZE = 2**8
LR = 0.001

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

class ImageDataset(torch.utils.data.Dataset):
    """
    Used to apply transforms to image part of numpy arrays
    """
    def __init__(self, data, targets, transform=None):
        super(ImageDataset, self).__init__()
        if transform is None:
            transform = lambda x: x
        self.data = torch.stack(tuple(transform(t) for t in data))
        self.targets = torch.LongTensor(targets)
        
    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)

def get_dataloaders(transform=None, **kwargs):
    if transform is None:
        transform = transforms.ToTensor()
    x_train, y_train, x_test, y_test = MNIST.get_data()

    train_data = ImageDataset(x_train, y_train, transform=transform)
    test_data = ImageDataset(x_test, y_test, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
    return train_loader, test_loader

def train(model, epoch, optimizer, criterion, dataloader, device):
    model.train()
    total_loss = 0
    for idx, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        optimizer.zero_grad()
        loss = criterion(y_pred, y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch} [{idx:04d}/{len(dataloader)}]: Average Loss = {round(total_loss/(idx), 4)}", end="\r")
    print("\n")

def test(model, epoch, criterion, dataloader, device):
    model.eval()
    total_loss = 0
    for idx, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = criterion(y_pred, y)
        total_loss += loss.item()
        print(f"Testing [{idx:04d}/{len(dataloader)}]: Average Loss = {round(total_loss/(idx), 4)}", end="\r")
    print("\n")

if __name__ == '__main__':
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE, shuffle=True)
    
    model = LeNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.95)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(EPOCHS):
        train(model, epoch, optimizer, criterion, train_loader, device)
        test(model, epoch, criterion, test_loader, device)
        print("\n\n")


