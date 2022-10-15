import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class TimeDistributed(nn.Module):
    def __init__(self, time_steps, layer, *args):
        super().__init__()

        self.layers = nn.ModuleList([layer(*args) for i in range(time_steps)])

    def forward(self, x):
        cur_device = x.device
        xsize = x.size()
        time_steps = xsize[1]
        output = torch.tensor([]).to(cur_device)
        # output = torch.tensor([]).to(device="cuda")
        for i in range(time_steps):
            if len(xsize) == 5:
                output_t = self.layers[i](x[:, i, :, :, :])
            if len(xsize) == 3:
                output_t = self.layers[i](x[:, i, :])
            try:
                output_t = output_t.unsqueeze(1)
            except:
                print("er")
            output = torch.cat((output, output_t), 1)
        return output


class Widar3(nn.Module):
    def __init__(self, time_steps, in_ch=1, classes=6) -> None:
        super().__init__()
        self.td_conv2d = TimeDistributed(time_steps, nn.Conv2d,  in_ch, 16, (3, 3))
        self.relu1 = TimeDistributed(time_steps, nn.ReLU)
        self.td_dropout1 = TimeDistributed(time_steps, nn.Dropout, 0.5)
        self.td_maxpl2d = TimeDistributed(time_steps, nn.MaxPool2d,  (2, 2))
        self.td_flatten = TimeDistributed(time_steps, nn.Flatten)
        self.td_dense1 = TimeDistributed(time_steps, nn.Linear,  1024, 64)
        self.relu2 = TimeDistributed(time_steps, nn.ReLU)
        self.td_dropout2 = TimeDistributed(time_steps, nn.Dropout, 0.7)
        self.td_dense2 = TimeDistributed(time_steps, nn.Linear,  64, 64)
        self.relu3 = TimeDistributed(time_steps, nn.ReLU)
        self.gru = nn.GRU(input_size=64, hidden_size=128, batch_first=True)
        self.dense3 = nn.Linear(128, classes)

    def forward(self, x):
        y = self.td_conv2d(x)
        y = self.relu1(y)
        y = self.td_dropout1(y)
        y = self.td_maxpl2d(y)
        y = self.td_flatten(y)
        y = self.td_dense1(y)
        y = self.relu2(y)
        y = self.td_dropout2(y)
        y = self.td_dense2(y)
        y = self.relu3(y)
        _, y = self.gru(y)
        y = y.squeeze(dim=0)
        y = F.dropout(y)
        y = self.dense3(y)
        return y


class RawWidar3(nn.Module):

    def __init__(self, time_steps, in_ch=1, classes=6) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            TimeDistributed(time_steps, nn.Conv2d,  in_ch, 16, (5, 5)),
            TimeDistributed(time_steps, nn.ReLU))
        # TimeDistributed(time_steps, nn.Dropout, 0.5))  # 16,16,16
        self.maxpl2d = TimeDistributed(time_steps, nn.MaxPool2d,  (2, 2))  # 32,8,8
        self.flatten = TimeDistributed(time_steps, nn.Flatten)
        self.dense1 = nn.Sequential(
            TimeDistributed(time_steps, nn.Linear,  1024, 64),
            TimeDistributed(time_steps, nn.ReLU),
            TimeDistributed(time_steps, nn.Dropout, 0.5))
        self.dense2 = nn.Sequential(
            TimeDistributed(time_steps, nn.Linear,  64, 64),
            TimeDistributed(time_steps, nn.ReLU))
        # TimeDistributed(time_steps, nn.Dropout, 0.5))
        self.gru = nn.GRU(input_size=64, hidden_size=128, batch_first=True)
        self.dense3 = nn.Linear(128, classes)

    def forward(self, x):
        y = self.conv1(x)
        y = self.maxpl2d(y)
        y = self.flatten(y)
        y = self.dense1(y)
        y = self.dense2(y)
        _, y = self.gru(y)
        y = y.squeeze(dim=0)
        y = F.dropout(y, 0.5)
        y = self.dense3(y)
        return y


class MyDeepWidar(nn.Module):
    def __init__(self, time_steps, in_ch=1, classes=6) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            TimeDistributed(time_steps, nn.Conv2d,  in_ch, 16, (3, 3)),
            TimeDistributed(time_steps, nn.ReLU),
            TimeDistributed(time_steps, nn.Dropout, 0.5))  # 16,18,18

        self.conv2 = nn.Sequential(
            TimeDistributed(time_steps, nn.Conv2d,  16, 32, (3, 3)),
            TimeDistributed(time_steps, nn.ReLU),
            TimeDistributed(time_steps, nn.Dropout, 0.5))  # 32,16,16
        self.maxpl2d = TimeDistributed(time_steps, nn.MaxPool2d,  (2, 2))  # 32,8,8

        self.flatten = TimeDistributed(time_steps, nn.Flatten)
        self.dense1 = nn.Sequential(
            TimeDistributed(time_steps, nn.Linear,  2048, 1024),
            TimeDistributed(time_steps, nn.ReLU),
            TimeDistributed(time_steps, nn.Dropout, 0.5))
        self.dense2 = nn.Sequential(
            TimeDistributed(time_steps, nn.Linear,  1024, 64),
            TimeDistributed(time_steps, nn.ReLU)
        )
        self.gru = nn.GRU(input_size=64, hidden_size=128, batch_first=True)
        self.dense3 = nn.Linear(128, classes)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.maxpl2d(y)
        y = self.flatten(y)
        y = self.dense1(y)
        y = self.dense2(y)
        _, y = self.gru(y)
        y = y.squeeze(dim=0)
        y = F.dropout(y)
        y = self.dense3(y)
        return y


class MyWidar(nn.Module):
    def __init__(self, time_steps, in_ch=1, classes=6) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            TimeDistributed(time_steps, nn.Conv2d,  in_ch, 16, (3, 3)),
            TimeDistributed(time_steps, nn.ReLU),
            TimeDistributed(time_steps, nn.Dropout, 0.5))  # 16,18,18

        self.conv2 = nn.Sequential(
            TimeDistributed(time_steps, nn.Conv2d,  16, 32, (3, 3)),
            TimeDistributed(time_steps, nn.ReLU),
            TimeDistributed(time_steps, nn.Dropout, 0.5))  # 32,16,16
        # self.maxpl2d = TimeDistributed(time_steps, nn.MaxPool2d,  (2, 2))  # 32,8,8

        self.flatten = TimeDistributed(time_steps, nn.Flatten)
        self.dense1 = nn.Sequential(
            TimeDistributed(time_steps, nn.Linear,  2048, 1024),
            TimeDistributed(time_steps, nn.ReLU),
            TimeDistributed(time_steps, nn.Dropout, 0.5))
        self.dense2 = nn.Sequential(
            TimeDistributed(time_steps, nn.Linear,  1024, 64),
            TimeDistributed(time_steps, nn.ReLU)
        )
        self.gru = nn.GRU(input_size=64, hidden_size=128, batch_first=True)
        self.dense3 = nn.Linear(128, classes)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.maxpl2d(y)
        y = self.flatten(y)
        y = self.dense1(y)
        y = self.dense2(y)
        _, y = self.gru(y)
        y = y.squeeze(dim=0)
        y = F.dropout(y)
        y = self.dense3(y)
        return y


class MyResWidar(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.res1 = torchvision.models.resnet18()

    def forward(self, x):
        y = x
        return y


if __name__ == "__main__":
    net = MyResWidar()
