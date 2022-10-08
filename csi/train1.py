import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

import loc_dataset

net = torchvision.models.resnet50()
train_dataset = loc_dataset.GesDataSet()
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
epochs = 2000
loss_fun = nn.MSELoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
best_loss = 1000.0

for epoch in range(epochs):
    net.train()
    sum_loss = 0.0
    for idx, (img, label) in enumerate(train_dataloader):
        img, label = torch.tensor(img), torch.tensor(img)
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
