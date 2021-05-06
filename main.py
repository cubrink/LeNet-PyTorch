import torch
import torch.nn as nn
from torchvision import transforms
import einops
from tqdm import tqdm

from models.models import LeNet
from utils.data import MNIST




### Hyperparams ###
EPOCHS = 5
BATCH_SIZE = 64

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:0'

# Load data
x_train, y_train, x_test, y_test = MNIST.get_data(format_='torch', device=device)


# Reshape image data from [batch, height, width, channel] to [batch, channel, height, width]
x_train = einops.rearrange(x_train, 'b h w c -> b c h w').to(device)
x_test = einops.rearrange(x_test, 'b h w c -> b c h w').to(device)

# Convert labels to ints
y_train = y_train.type(torch.LongTensor).to(device)
y_test = y_test.type(torch.LongTensor).to(device)

# Format dataset, create dataloader
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


# Define model, optimizer and loss function
lenet = LeNet().to(device)
optimizer = torch.optim.SGD(lenet.parameters(), lr=0.001)
loss_func = torch.nn.CrossEntropyLoss().to(device)

# Prep for training
lenet.train()

# for X, y in train_loader:
#     print(X)
#     print(y)
#     print(lenet(X))
#     break


for epoch in range(EPOCHS):
    losses = []
    for X, y in tqdm(train_loader, desc='Training batches'):
        X = (X * 255)*2 - 1
        optimizer.zero_grad()
        y_pred = lenet(X.to(device))
        loss = loss_func(y_pred, y)
        losses.append(loss)
        loss.backward()
        optimizer.step()
    total_loss = sum(loss.item() for loss in losses)
    print(f"Epoch [{epoch+1}/{EPOCHS}]: Loss = {round(total_loss, 4)}")

    
