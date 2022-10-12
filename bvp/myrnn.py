import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDistributed(nn.Module):
    def __init__(self, time_steps, layer, *args):
        super().__init__()

        self.layers = nn.ModuleList([layer(*args) for i in range(time_steps)])

    def forward(self, x):
        xsize = x.size()
        time_steps = xsize[1]
        output = torch.tensor([]).to(device="cuda")
        for i in range(time_steps):
            if len(xsize) == 5:
                output_t = self.layers[i](x[:, i, :, :, :])
            if len(xsize) == 3:
                output_t = self.layers[i](x[:, i, :])
            try:
                output_t = output_t.unsqueeze(1).to(device="cuda")
            except:
                print("er")
            output = torch.cat((output, output_t), 1)
        return output


class Widar3(nn.Module):
    def __init__(self, time_steps, in_ch=1, classes=6) -> None:
        super().__init__()
        self.td_conv2d = TimeDistributed(time_steps, nn.Conv2d,  in_ch, 16, (5, 5))
        self.td_maxpl2d = TimeDistributed(time_steps, nn.MaxPool2d,  (2, 2))
        self.td_flatten = TimeDistributed(time_steps, nn.Flatten)
        self.td_dense1 = TimeDistributed(time_steps, nn.Linear,  1024, 64)
        self.td_dropout1 = TimeDistributed(time_steps, nn.Dropout)
        self.td_dense2 = TimeDistributed(time_steps, nn.Linear,  64, 64)
        self.gru = nn.GRU(input_size=64, hidden_size=128, batch_first=True)
        self.dense3 = nn.Linear(128, classes)

    def forward(self, x):
        y = F.relu(self.td_conv2d(x))
        y = self.td_maxpl2d(y)
        y = self.td_flatten(y)
        y = F.relu(self.td_dense1(y))
        y = self.td_dropout1(y)
        y = F.relu(self.td_dense2(y))
        _, y = self.gru(y)
        y = y.squeeze(dim=0)
        y = F.dropout(y)
        y = self.dense3(y)
        return y


if __name__ == "__main__":
    import numpy as np
    test_data = torch.rand(64, 25, 1, 20, 20)

    net = Widar3(time_steps=25)
    y = net(test_data)
