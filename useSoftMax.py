import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.utils.data as Data
import torch.nn as nn
from torch import optim
from torch.nn import functional as F

data_set = torchvision.datasets.MNIST(root='./mnist/', transform=torchvision.transforms.ToTensor())
#print(data_set.data.size())     #torch.Size([60000, 28, 28])
#print(data_set.targets.size())      #torch.Size([60000])
data_loader = Data.DataLoader(dataset=data_set, batch_size=150, shuffle=True)    #image batch size shape = (150, 1, 28, 28)
#2000 samples for testing
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)/255     #shape from (2000, 28, 28)to(2000, 1, 28, 28)
test_y = test_data.targets

'''
#plot one eg
plt.imshow(data_set.data[0].numpy(), cmap='gray')
plt.title('%i' % data_set.targets[0])
plt.show()
'''


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc(x)
        x = F.softmax(x)
        return x


lor = LogisticRegression()
#print(lor)
opt = torch.optim.Adam(lor.parameters(), 0.001)
loss_func = nn.CrossEntropyLoss()

#training and testing
for epoch in range(20):
    for step, (b_x, b_y) in enumerate(data_loader):
        output = lor(b_x)
        loss = loss_func(output, b_y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 50 == 0:
            test_output = lor(test_x)
            predicted_y = torch.max(test_output, 1)[1].numpy()
            acc = float((predicted_y == test_y.numpy()).astype(int).sum())/float(test_y.size(0))
            print('Epoch:', epoch, '|train loss:%.4f' % loss.data.numpy(), '|test acc:%.2f' % acc)


test_output = lor(test_x[:20])
predicted_y = torch.max(test_output, 1)[1].numpy()
print(predicted_y, 'prediction number')
print(test_y[:20].numpy(), 'real number')

'''
Epoch: 19 |train loss:1.5496 |test acc:0.93
[7 2 1 0 4 1 4 9 6 9 0 6 9 0 1 5 9 7 3 4] prediction number
[7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4] real number
'''
