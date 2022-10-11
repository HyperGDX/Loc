import os

import pandas as pd
import scipy.io as scio
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch.nn.functional as F


def get_mat_kind(path: str):
    mat_info = path.split("-")
    mat_dict = dict()
    mat_dict["userid"] = int(mat_info[0].split("user")[1])
    mat_dict["ges_id"] = int(mat_info[1])
    mat_dict["loc_id"] = int(mat_info[2])
    mat_dict["face_id"] = int(mat_info[3])
    mat_dict["rep_id"] = int(mat_info[4])
    return mat_dict


class BVPDataSet(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data_dir = "data/20181109-VS/6-link/user1"
        self.path_lst = os.listdir(self.data_dir)

    def __getitem__(self, index):
        data_dict = dict()
        cur_path = self.path_lst[index]
        label_dict = get_mat_kind(cur_path)
        cur_mat = scio.loadmat(os.path.join(self.data_dir, cur_path))['velocity_spectrum_ro']
        cur_mat = np.transpose(cur_mat, axes=[2, 0, 1])
        data_dict["bvp"] = torch.from_numpy(cur_mat)
        data_dict["Tlength"] = data_dict["bvp"].shape[2]
        # data_dict = {**data_dict, **label_dict}
        data_dict["ges_id"] = F.one_hot(torch.tensor(label_dict["ges_id"]-1), 6)
        return data_dict

    def __len__(self):
        return len(self.path_lst)


def collate_func(batch_dic):

    fea_batch = []
    label_batch = []
    len_batch = []
    for i in range(len(batch_dic)):
        dic = batch_dic[i]
        fea_batch.append(dic['bvp'])
        label_batch.append(dic['ges_id'])
        len_batch.append(dic['Tlength'])
        # mask_batch[i, :dic['Tlength']] = 1  # mask
    res = {}
    res['bvp'] = pad_sequence(fea_batch, batch_first=True)  # 将信息封装在字典res中
    res['Tlength'] = len_batch
    # res['ges_id'] = pad_sequence(label_batch, batch_first=True)
    res['ges_id'] = label_batch
    # res['id'] = id_batch
    # res['mask'] = mask_batch
    return res


if __name__ == "__main__":
    ds = BVPDataSet()
    dl = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_func)
    for i in dl:
        print(i.keys())
