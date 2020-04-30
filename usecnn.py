import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.utils.data as Data
import torch.nn as nn
from torch import optim
from torch.nn import functional as F

EPOCH = 5
BATCH_SIZE = 50
LR = 0.001

train_set = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=torchvision.transforms.ToTensor())
train_loader = Data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)

test_set = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = torch.unsqueeze(test_set.data, dim=1).type(torch.FloatTensor)[:2000]/255
test_y = test_set.targets[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))    #max pooling with a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x)
        return x


cnn = CNN()

optimizer = optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        out = cnn(b_x)
        loss = loss_func(out, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_out = cnn(test_x)
            predicted_y = torch.max(test_out, 1)[1].data.numpy()
            acc = float(((predicted_y == test_y.numpy())).astype(int).sum())/float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % acc)


test_out = cnn(test_x[:30])
pred_y = torch.max(test_out, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:30].numpy(), 'real number')