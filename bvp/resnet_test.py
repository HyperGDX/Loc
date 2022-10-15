import torch
import torchvision
import torch.nn as nn


class My3dconv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # B*C*H*W*T
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(5, 5, 1))

    def forward(self, x):
        y = self.conv1(x)
        return y


if __name__ == "__main__":
    net = My3dconv()
    x = torch.randn((32, 1, 20, 20, 25))

    y = net(x)
    print(y.shape)
