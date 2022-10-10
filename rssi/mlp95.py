import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dataset95
import numpy as np

net = nn.Sequential(
    nn.Linear(9, 16),
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
    nn.Linear(16, 2)
)

train_dataset = dataset95.Dataset95()
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
EPOCH = 2000
loss_fun = nn.MSELoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

train_loss = []
valid_loss = []
train_epochs_loss = []
valid_epochs_loss = []

for epoch in range(EPOCH):
    net.train()
    train_epoch_loss = []
    for idx, (data_x, data_y) in enumerate(train_dataloader, 0):
        data_x = data_x.to(torch.float32).to(device)
        data_y = data_y.to(torch.float32).to(device)
        outputs = net(data_x)
        optimizer.zero_grad()
        loss = loss_fun(data_y, outputs)
        loss.backward()
        optimizer.step()
        train_epoch_loss.append(loss.item())
        train_loss.append(loss.item())
        # if idx % (len(train_dataloader)//2) == 0:
        # print(f"epoch={epoch}/{EPOCH},{idx}/{len(train_dataloader)}of train, loss={loss.item()}, lr={optimizer.state_dict()['param_groups'][0]['lr']}")
        if idx % (len(train_dataloader)) == 0:
            print(f"epoch={epoch}/{EPOCH}, loss={loss.item()}, lr={optimizer.state_dict()['param_groups'][0]['lr']}")

    scheduler.step()
    train_epochs_loss.append(np.average(train_epoch_loss))
