import read_csi
import numpy as np
import pandas as pd
import os

TOTAL_RECEIVER = 6
TOTAL_GES = 3
TOTAL_LOC = 5
TOTAL_FACE = 5
TOTAL_REPEAT = 20


# def padding(nd):
#     if nd.shape[0] < 2000:
#         nd = np.con
#     else:
#         nd = nd[:2000]
#     return nd


# new_nd = read_csi.get_csi_data(r"csi\widar_data\csi\user1_1\user1-1-1-1-1-r2.dat")
# print(new_nd.shape)
# new_nd = padding(new_nd)
# print(new_nd.shape)

def read_y_from_path(path: str):
    i = path.split("user")[-1]
    ys = i.split("-")
    ges = ys[1]
    loc = ys[2]
    face = ys[3]
    return [ges, loc, face]


# path_list = os.listdir("csi/widar_data/csi/user1_1")
# raw_data = pd.DataFrame(columns=["ges", "loc", "face", "csi_data"])
# for p in path_list:
#     a = read_y_from_path(p)
#     print(a)
#     new_nd = read_csi.get_csi_data("csi/widar_data/csi/user1_1/"+p)[:1900, :, :, :]
#     new_nd = np.expand_dims(new_nd, axis=0)
#     for i in range(1, TOTAL_RECEIVER):
#         cur_nd = read_csi.get_csi_data(f"csi/widar_data/csi/user1_1/user1-1-1-1-1-r{i+1}.dat")[:1900, :, :, :]
#         cur_nd = np.expand_dims(cur_nd, axis=0)
#         new_nd = np.concatenate((new_nd, cur_nd), axis=0)
#     cur_df = pd.DataFrame(data=read_y_from_path(p).append(new_nd), columns=["ges", "loc", "face", "csi_data"])
#     raw_data = pd.concat([raw_data, cur_df], ignore_index=True)
#     print(raw_data)
