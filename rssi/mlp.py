import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import mydataset

net = nn.Sequential(
    nn.Linear(1, 16),
    nn.RReLU(),
    nn.Linear(16, 32),
    nn.RReLU(),
    nn.Linear(32, 64),
    nn.RReLU(),
    nn.Linear(64, 64),
    nn.RReLU(),
    nn.Linear(64, 32),
    nn.RReLU(),
    nn.Linear(32, 16),
    nn.RReLU(),
    nn.Linear(16, 1)
)
train_dataset = mydataset.A_N_DataSet()
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
epochs = 2000
loss_fun = nn.MSELoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
best_loss = 1000.0

for epoch in range(epochs):
    net.train()
    sum_loss = 0.0
    for idx, (img, label) in enumerate(train_dataloader):
        img, label = torch.unsqueeze(img, dim=1), torch.unsqueeze(label, dim=1)
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        output = net(img)
        train_loss = loss_fun(output, label)
        train_loss.backward()
        optimizer.step()
        sum_loss += train_loss.item()
    final_loss = sum_loss / idx
    if final_loss < best_loss:
        best_loss = final_loss
        torch.save(net.state_dict(), 'best_net_params.pth')
    print('[%d] Train loss:%.09f' % (epoch, final_loss))
