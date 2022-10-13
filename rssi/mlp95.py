import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import dataset95
import numpy as np


model = nn.Sequential(
    nn.Linear(9, 16),
    nn.ReLU(),
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 2)
)


def get_train_device():
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    elif torch.cuda.is_available():
        device = torch.device("cuda")
    return device


device = get_train_device()

full_dataset = dataset95.Dataset95()
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256)
EPOCH = 2000
loss_fun = nn.MSELoss()

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

train_loss = []
valid_loss = []
train_epochs_loss = []
valid_epochs_loss = []

for epoch in range(EPOCH):
    model.train()
    train_epoch_loss = []
    for idx, (data_x, data_y) in enumerate(train_dataloader, 0):
        data_x = data_x.to(torch.float32).to(device)
        data_y = data_y.to(torch.float32).to(device)
        outputs = model(data_x)
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

    train_epochs_loss.append(np.average(train_epoch_loss))
    # #### validation ####
    # model.eval()

    # size = len(test_loader.dataset)
    # num_batches = len(test_loader)
    # test_loss, correct = 0, 0
    # with torch.no_grad():
    #     for img, label in test_loader:
    #         data_x = img.to(torch.float32).to(device)
    #         data_y = label.to(torch.float32).to(device)
    #         pred = model(data_y)
    #         test_loss += loss_fun(pred, label).item()

    # print(f"Test loss: {np.sqrt(test_loss):>8f}")

    # scheduler.step()
