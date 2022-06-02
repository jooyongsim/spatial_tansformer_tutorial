# # This is a tutorial for spatial transformer in Sookmyung Wonmen's Univ's
# Deep Learning Course 2022 with Prof. Joo Yong Sim
# Source: https://github.com/pytorch/tutorials/blob/master/advanced_source/neural_style_tutorial.py

from torchvision import datasets, transforms
from torchvision.utils import make_grid

import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import numpy as np

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307),(0.3081))
])
train_loader = DataLoader(datasets.MNIST(root = '.', train = True, \
                    transform = transform, download=True),\
                    batch_size=64, shuffle = True)

test_loader = DataLoader(datasets.MNIST(root = '.', train = False, \
                    transform = transform, download=True),\
                    batch_size=64, shuffle = True)


class STN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, 7),
            nn.MaxPool2d(2,2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, 5),
            nn.MaxPool2d(2,2),
            nn.ReLU(True),
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10*3*3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3*2),
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,0,1,0], \
                                                    dtype=torch.float))
    def forward(self,x):
        xs = self.localization(x)
        xs = xs.reshape(-1,10*3*3)
        theta = self.fc_loc(xs)
        theta = theta.reshape(-1,2,3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x, theta, grid

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stn = STN()

        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50)
        self.fc1_drop = nn.Dropout()
        self.fc2 = nn.Linear(50,10)
        
    def forward(self, x):
        x, theta, grid = self.stn(x)
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x = x.reshape(-1,320)
        x = self.fc1_drop(self.fc1(x))
        x = self.fc2(x)
        return x # F.log_softmax(x, dim=1)

device = torch.device('cuda')
model = Net().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr = 0.01)
loss_func = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    for step, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        output = model(X)
        loss = loss_func(output, y)
        #  loss = F.nll_loss(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step%100 == 0:
            print(f'Train Epoch: {epoch} [{step*len(X):5d}/{len(train_loader.dataset)} \
            Loss: {loss.item():.6f}')
def test():
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            test_loss += loss_func(output, y).item()
            # test_loss += F.nll_loss(output, y, size_average=False).item()

            pred = torch.argmax(output, dim=1)
            correct += pred.eq(y).sum().item()
        test_loss /= len(test_loader.dataset)
        print(f'Test set: Avg loss is {test_loss:.4f}, Accuracy is {correct:5d\
            } /{len(test_loader.dataset)} ({100.*correct / len(test_loader.dataset):.0f}%')

for epoch in range(1, 21):
    train(epoch)
    test()
    
def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.
def visualize_stn():
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data)[0].cpu()

        in_grid = convert_image_np(
            make_grid(input_tensor))

        out_grid = convert_image_np(
            make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')

# Visualize the STN transformation on some input batch
visualize_stn()

plt.ioff()
plt.show()
