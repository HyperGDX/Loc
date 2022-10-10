import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class Dataset95(Dataset):
    def __init__(self) -> None:
        super().__init__()
        raw_df = pd.read_csv(r"rssi\data\9.5\2022_9_5_bt_data_static.csv", header=0)
        self.data_df = raw_df[["rel_x", "rel_y",
                               "rssi1", "rssi2", "rssi3", "rssi4",
                               "rssi5", "rssi6", "rssi7", "rssi8", "rssi9"]]
        self.data_df = self.data_df.fillna(value=-100)
        self.loc_df = self.data_df[["rel_x", "rel_y"]]
        self.rssi_df = self.data_df.drop(["rel_x", "rel_y"], axis=1)
        self.rssi_df = (self.rssi_df - self.rssi_df.mean())/self.rssi_df.std()

    def __getitem__(self, index):
        x = self.rssi_df.iloc[index].to_numpy().astype(np.float32)
        y = self.loc_df.iloc[index].to_numpy().astype(np.float32)
        return torch.tensor(x), torch.tensor(y)

    def __len__(self):
        return self.data_df.shape[0]
