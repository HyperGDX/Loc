import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import myrnn
import read_bvp

ALL_MOTION = [1, 2, 3, 4, 5, 6]
N_MOTION = len(ALL_MOTION)
batch_size = 1024

full_dataset = read_bvp.BVPDataSet(data_dir="data/BVP", motion_sel=ALL_MOTION)
train_size = int(0.9 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=8)


def get_train_device():
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    elif torch.cuda.is_available():
        device = torch.device("cuda")
    return device


device = get_train_device()
EPOCH = 1000
TIME_STEPS = full_dataset.get_T_max()


# model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
model = myrnn.RawWidar3(time_steps=TIME_STEPS, in_ch=1, classes=6)
model.to(device)


criterion = nn.CrossEntropyLoss()

learning_rate = 0.01

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 100, 300, 500], gamma=0.1)

for epoch in range(EPOCH):
    #### train ####
    model.train()
    sum_loss = 0.0
    total_correct = 0
    total_sample = 0
    for idx, data in enumerate(train_loader):
        # img, label = torch.unsqueeze(img, dim=1), torch.unsqueeze(label, dim=1)
        img, label = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(img)
        train_loss = criterion(output, label)
        train_loss.backward()
        optimizer.step()
        real_label = torch.max(label.data, 1)[1]
        predicted = torch.max(output.data, 1)[1]
        total_correct += (predicted == real_label).sum()
        sum_loss += train_loss.item()
    final_loss = sum_loss / idx

    print(f"Epoch [{epoch}] lr: {optimizer.state_dict()['param_groups'][0]['lr']}", end=" ")
    print('Train loss:', final_loss, end=" ")
    print('Train acc:', (total_correct.item())/len(train_dataset)*100, end=" ")
    scheduler.step()

    #### validation ####
    model.eval()

    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for img, label in test_loader:
            img, label = data[0].to(device), data[1].to(device)
            pred = model(img)
            test_loss += criterion(pred, label).item()
            correct += (pred.argmax(1) == label.argmax(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
