import os
import scipy.io as scio
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import scipy


ALL_MOTION = [1, 2, 3, 4, 5, 6]
N_MOTION = len(ALL_MOTION)


class BVPDataSet(Dataset):
    def __init__(self, data_dir, motion_sel) -> None:
        super().__init__()
        data_name_lst = self.find_all_files(data_dir, ".mat")
        self.data, self.label, self.t_max = self.load_data(data_name_lst, motion_sel)

    def __getitem__(self, index):
        x = self.data[index].astype(np.float32)
        y = self.label[index]
        return x, y

    def __len__(self):
        return len(self.label)

    def normalize_data(self, data_1):
        # data(ndarray)=>data_norm(ndarray): [20,20,T]=>[20,20,T]
        data_1_max = np.concatenate((data_1.max(axis=0), data_1.max(axis=1)), axis=0).max(axis=0)
        data_1_min = np.concatenate((data_1.min(axis=0), data_1.min(axis=1)), axis=0).min(axis=0)
        if (len(np.where((data_1_max - data_1_min) == 0)[0]) > 0):
            return data_1
        data_1_max_rep = np.tile(data_1_max, (data_1.shape[0], data_1.shape[1], 1))
        data_1_min_rep = np.tile(data_1_min, (data_1.shape[0], data_1.shape[1], 1))
        data_1_norm = (data_1 - data_1_min_rep) / (data_1_max_rep - data_1_min_rep)
        return data_1_norm

    def zero_padding(self, data, T_MAX):
        # data(list)=>data_pad(ndarray): [20,20,T1/T2/...]=>[20,20,T_MAX]
        data_pad = []
        for i in range(len(data)):
            t = np.array(data[i]).shape[2]
            data_pad.append(np.pad(data[i], ((0, 0), (0, 0), (T_MAX - t, 0)), 'constant', constant_values=0).tolist())
        return np.array(data_pad)

    def onehot_encoding(self, label, num_class):
        # label(list)=>_label(ndarray): [N,]=>[N,num_class]
        label = np.array(label).astype('int32')
        # assert (np.arange(0,np.unique(label).size)==np.unique(label)).prod()    # Check label from 0 to N
        label = np.squeeze(label)
        _label = np.eye(num_class)[label-1]     # from label to onehot
        return _label

    def load_data(self, data_name_lst, motion_sel):
        print("get data_name_lst finish")
        t_max = 0
        data = []
        label = []
        data_len = len(data_name_lst)
        for idx, file_path in enumerate(data_name_lst):
            if idx % 100 == 0:
                print(f"Load {round(idx/data_len*100,2)}%")
            if idx == data_len-1:
                print("Load all mat")
            # data/20181109-VS/6-link/user3/user3-1-1-1-1-1-1e-07-100-20-100000-L0.mat
            data_file_name = file_path.split("/")[-1]
            try:

                data_1 = scio.loadmat(file_path)['velocity_spectrum_ro']
                label_1 = int(data_file_name.split('-')[1])
                # location = int(data_file_name.split('-')[2])
                # orientation = int(data_file_name.split('-')[3])
                # repetition = int(data_file_name.split('-')[4])

                # Select Motion
                if (label_1 not in motion_sel):
                    continue

                # Select Location
                # if (location not in [1,2,3,5]):
                #     continue

                # Select Orientation
                # if (orientation not in [1,2,4,5]):
                #     continue

                # Normalization
                data_normed_1 = self.normalize_data(data_1)

                # Update T_MAX
                if t_max < np.array(data_1).shape[2]:
                    t_max = np.array(data_1).shape[2]
            except scipy.io.matlab.miobase.MatReadError:
                print(f"{file_path} has no data")
                continue
            # except scipy.io.matlab._miobase.MatReadError:
            #     print(f"{file_path} has no data")
            #     continue
            except IndexError:
                print(f"{file_path} Index error")
                continue

                # Save List
            data.append(data_normed_1.tolist())
            label.append(label_1)

        # Zero-padding
        data = self.zero_padding(data, t_max)

        # Swap axes
        data = np.swapaxes(np.swapaxes(data, 1, 3), 2, 3)   # [N,20,20',T_MAX]=>[N,T_MAX,20,20']
        data = np.expand_dims(data, axis=2)    # [N,T_MAX,20,20]=>[N,T_MAX,1,20,20]

        # Convert label to ndarray
        label = np.array(label)
        label_onehot = np.eye(6)[label-1]
        # data(ndarray): [N,T_MAX,20,20,1], label(ndarray): [N,N_MOTION]
        print("load all data finish")
        return data, label_onehot, t_max

    def get_T_max(self):
        return self.t_max

    def find_all_files(self, path, file_type):    # ??????path???????????????file_type??????????????????????????????
        f_list = []

        def files_list(father_path):
            sub_path = os.listdir(father_path)    # ????????????????????????????????????????????????
            for sp in sub_path:
                full_sub_path = "/".join([father_path, sp])    # ?????????????????????
                if os.path.isfile(full_sub_path):    # ?????????????????????
                    file_name, post_name = os.path.splitext(full_sub_path)    # ?????????????????????
                    if post_name == file_type:
                        f_list.append(file_name + post_name)
                else:    # ?????????????????????????????????
                    files_list(full_sub_path)
        files_list(path)
        return f_list


if __name__ == "__main__":
    # ds = BVPDataSet(path_to_data="data/20181109-VS/6-link/user1", motion_sel=ALL_MOTION)
    ds = BVPDataSet(data_dir="data/BVP", motion_sel=ALL_MOTION)
    print(len(ds))
