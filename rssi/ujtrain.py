from ujdataset import UJDataSet
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torchvision.transforms as transforms


net = nn.Sequential(
    nn.Linear(520, 256),
    nn.ReLU(),
    nn.Dropout(0.7),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.8),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)
train_dataset = UJDataSet("train")
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
epochs = 2000
loss_fun = nn.CrossEntropyLoss()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "mps"
net = net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
best_loss = 1000.0

for epoch in range(epochs):
    net.train()
    sum_loss = 0.0
    sum_acc = 0.0
    for idx, (img, label) in enumerate(train_dataloader):
        # img, label = torch.unsqueeze(img, dim=1), torch.unsqueeze(label, dim=1)
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        output = net(img)
        train_loss = loss_fun(output, label)
        train_loss.backward()
        optimizer.step()
        sum_loss += train_loss.item()
        print(output)

        # orrect += (predicted == y_batch).sum()
    final_loss = sum_loss / idx
    if final_loss < best_loss:
        best_loss = final_loss
        torch.save(net.state_dict(), 'best_net_params.pth')
    print(f"Epoch {epoch} lr: {optimizer.state_dict()['param_groups'][0]['lr']}", end=" ")
    print('[%d] Train loss:%.09f' % (epoch, final_loss))
    scheduler.step()
