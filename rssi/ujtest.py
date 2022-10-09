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
test_dataset = UJDataSet()
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "mps"
net = net.to(device)


net.train()
acc = 0.0
for idx, (img, label) in enumerate(test_dataloader):
    # img, label = torch.unsqueeze(img, dim=1), torch.unsqueeze(label, dim=1)
    img, label = img.to(device), label.to(device)
    output = net(img)

    sum_loss += train_loss.item()
final_loss = sum_loss / idx
if final_loss < best_loss:
    best_loss = final_loss
    torch.save(net.state_dict(), 'best_net_params.pth')
print(f"Epoch {epoch} lr: {optimizer.state_dict()['param_groups'][0]['lr']}", end=" ")
print('[%d] Train loss:%.09f' % (epoch, final_loss))
