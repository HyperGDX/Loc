import torch
import torch
from torch.utils.data import Dataset
import read_csi
import os

# csi\widar_data\csi\user1_1
total_user = 1
total_face = 5
total_repeat = 20


total_ges = 3
total_loc = 5


total_receiver = 6


# def padding(nd):
#     return


def read_y_from_path(path: str):
    i = path.split("user")[-1]
    ys = i.split("-")
    ges = ys[1]
    loc = ys[2]
    face = ys[3]
    receiver = ys[5]
    return [ges, loc, face, receiver]


class GesDataSet(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.path_list = os.listdir("csi/widar_data/csi/user1_1")

    def __getitem__(self, index):
        cur_path = self.path_list[index]
        x = read_csi.get_csi_data("csi/widar_data/csi/user1_1/"+cur_path)[:1000, :, :, :]
        y = read_y_from_path(cur_path)
        return x, y

    def __len__(self):
        return len(self.path_list)
