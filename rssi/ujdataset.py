import torch
from torch.utils.data import Dataset
import pandas as pd


class UJDataSet(Dataset):
    def __init__(self, kind) -> None:
        super().__init__()
        if kind == "train":
            self.raw_data = pd.read_csv(r"rssi/data/UJIndoorLoc/trainingData.csv")
        else:
            self.raw_data = pd.read_csv(r"rssi/data/UJIndoorLoc/validationData.csv")

    def __getitem__(self, index):
        x = self.raw_data.iloc[index, 0:520]
        x = (x-x.min())/(x.max()-x.min())
        # df = (df-df.min())/(df.max()-df.min())
        y = self.raw_data.iloc[index, 522:525]
        # y = (y-y.min())/(y.max()-y.min())
        return torch.tensor(x).to(torch.float32), torch.tensor(y).to(torch.float32)

    def __len__(self):
        return self.raw_data.shape[0]
