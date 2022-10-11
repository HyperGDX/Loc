import gru_net_test
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import Tensor
from torch.autograd import Variable
from torch.nn import Parameter
from torch.utils.data import Dataset, DataLoader
import read_bvp


batch_size = 128
train_dataset = read_bvp.BVPDataSet()
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=read_bvp.collate_func)
device = "cuda" if torch.cuda.is_available() else "cpu"

EPOCH = 1000


# model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
model = gru_net_test.CustomRNN(input_size=20, hidden_size=128, num_layers=2, batch_first=True)
model.to(device)


criterion = nn.CrossEntropyLoss()

learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)


for epoch in range(EPOCH):
    model.train()
    sum_loss = 0.0
    sum_acc = 0.0
    for idx, (img, label) in enumerate(train_loader):
        # img, label = torch.unsqueeze(img, dim=1), torch.unsqueeze(label, dim=1)
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(img)
        train_loss = criterion(output, label)
        train_loss.backward()
        optimizer.step()
        sum_loss += train_loss.item()
    final_loss = sum_loss / idx
    # if final_loss < best_loss:
    #     best_loss = final_loss
    #     torch.save(model.state_dict(), 'best_net_params.pth')
    print(f"Epoch {epoch} lr: {optimizer.state_dict()['param_groups'][0]['lr']}", end=" ")
    print('[%d] Train loss:%.09f' % (epoch, final_loss))
    scheduler.step()

    # if iter % 500 == 0:
    #     # Calculate Accuracy
    #     correct = 0
    #     total = 0
    #     # Iterate through test dataset
    #     for images, labels in test_loader:

    #         if torch.cuda.is_available():
    #             images = Variable(images.view(-1, seq_dim, input_dim).cuda())
    #         else:
    #             images = Variable(images.view(-1, seq_dim, input_dim))

    #         # Forward pass only to get logits/output
    #         outputs = model(images)

    #         # Get predictions from the maximum value
    #         _, predicted = torch.max(outputs.data, 1)

    #         # Total number of labels
    #         total += labels.size(0)

    #         # Total correct predictions
    #         #######################
    #         #  USE GPU FOR MODEL  #
    #         #######################
    #         if torch.cuda.is_available():
    #             correct += (predicted.cpu() == labels.cpu()).sum()
    #         else:
    #             correct += (predicted == labels).sum()

    #     accuracy = 100 * correct / total

    #     # Print Loss
    #     print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))
