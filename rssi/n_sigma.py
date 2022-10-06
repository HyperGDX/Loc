import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# def n_sigma_ser(Ser1, n):
#     rule = ((Ser1.mean()-n*Ser1.std()) > Ser1) | ((Ser1.mean()+n*Ser1.std()) < Ser1)
#     index = np.arange(Ser1.shape[0])[rule]
#     return index  # 返回落在3sigma之外的行索引值


# def delete_out3sigma(data, n):
#     out_index = []  # 保存要删除的行索引
#     index = n_sigma_ser(data.iloc[:, 2], n)
#     out_index += index.tolist()
#     delete_ = list(set(out_index))
#     print('所删除的行索引为：', delete_)
#     data.drop(index=delete_, inplace=True)
#     return data


def __n_sigma_main(df: pd.DataFrame, time_start, time_stop, n=1, direct_use=False):
    first_time = df.loc[0][0]
    time_start = first_time+time_start
    time_stop = time_start+time_stop
    means = []
    stds = []
    if direct_use == True:
        ans_df = pd.DataFrame(columns=["time", "d", "rssi"])
        for i in [1, 3, 6, 9]:
            filter_df = df.loc[(df['time'] >= time_start) & (df['time'] <= time_stop) & (df['d'] == i)]
            print(f"before n sigma, {i} data num: {filter_df.shape[0]}", end=" ")
            cur_rssi = filter_df["rssi"]
            means.append(cur_rssi.mean())
            stds.append(n*cur_rssi.std())
            rule = ((cur_rssi.mean()-n*cur_rssi.std()) < cur_rssi) & ((cur_rssi.mean()+n*cur_rssi.std()) > cur_rssi)
            after_df = filter_df.loc[rule]
            print(f" after n sigma, {i} data num: {after_df.shape[0]}")
            ans_df = pd.concat([ans_df, after_df], ignore_index=True)
        return ans_df, means, stds
    else:
        rssi_lst = []
        for i in [1, 3, 6, 9]:
            filter_df = df.loc[(df['time'] >= time_start) & (df['time'] <= time_stop) & (df['d'] == i)]
            print(f"before n sigma, {i} data num: {filter_df.shape[0]}", end=" ")
            cur_rssi = filter_df["rssi"]
            rule = ((cur_rssi.mean()-n*cur_rssi.std()) < cur_rssi) & ((cur_rssi.mean()+n*cur_rssi.std()) > cur_rssi)
            after_df = filter_df.loc[rule]
            print(f" after n sigma, {i} data num: {after_df.shape[0]}")
            rssi_lst.append(after_df)
        return rssi_lst


def n_sigma(df: pd.DataFrame, time_start, time_stop, n, direct_use=False):
    print(f"raw data has {df.shape[0]} data")
    print(f"select data from {time_start} second to {time_stop} second with n={n}")
    if direct_use == True:
        n_sigma_df_after, _, _ = __n_sigma_main(df, time_start, time_stop, n, direct_use)
        print(f"after has {n_sigma_df_after.shape[0]} data left")
        return n_sigma_df_after
    else:
        return __n_sigma_main(df, time_start, time_stop, n, direct_use)


if __name__ == "__main__":
    raw_data = pd.read_csv("rssi/data/rssi_d/rssi_d.csv")
    d1369 = n_sigma(raw_data, time_start=0, time_stop=60, n=1.5, direct_use=False)
