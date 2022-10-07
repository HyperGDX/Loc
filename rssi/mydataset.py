import torch
from torch.utils.data import Dataset

import read_data

d_r_nd = read_data.d_r_nd


class A_N_DataSet(Dataset):

    def __getitem__(self, index):
        x = d_r_nd[0, index]
        y = d_r_nd[1, index]
        return torch.tensor(x), torch.tensor(y)

    def __len__(self):
        return len(d_r_nd[0])
