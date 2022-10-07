import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def n_sigma(Ser1, n):
    rule = ((Ser1.mean()-n*Ser1.std()) > Ser1) | ((Ser1.mean()+n*Ser1.std()) < Ser1)
    index = np.arange(Ser1.shape[0])[rule]
    return index  # 返回落在3sigma之外的行索引值


def delete_out3sigma(data, n):
    out_index = []  # 保存要删除的行索引
    index = n_sigma(data.iloc[:, 2], n)
    out_index += index.tolist()
    delete_ = list(set(out_index))
    print('所删除的行索引为：', delete_)
    data.drop(index=delete_, inplace=True)
    return data


def n_sigma_df(df: pd.DataFrame, time_start=0, time_stop=30, n=1):
    first_time = df.loc[0][0]
    time_start = first_time+time_start
    time_stop = time_start+time_stop
    means = []
    stds = []
    ans_df = pd.DataFrame(columns=["time", "d", "rssi"])
    for i in [1, 3, 6, 9]:
        filter_df = df.loc[(df['time'] >= time_start) & (df['time'] <= time_stop) & (df['d'] == i)]
        cur_rssi = filter_df["rssi"]
        means.append(cur_rssi.mean())
        stds.append(n*cur_rssi.std())
        rule = ((cur_rssi.mean()-n*cur_rssi.std()) < cur_rssi) & ((cur_rssi.mean()+n*cur_rssi.std()) > cur_rssi)
        after_df = filter_df.loc[rule]
        ans_df = pd.concat([ans_df, after_df], ignore_index=True)
    return ans_df, means, stds


if __name__ == "__main__":
    raw_data = pd.read_csv("rssi/data/rssi_d/rssi_d.csv")
    time_start = 0
    time_stop = 60
    n = 1.5
    b, ms, ss = n_sigma_df(raw_data, time_start, time_stop, n)

    first_time = raw_data.loc[0][0]
    time_start = first_time+time_start
    time_stop = time_start+time_stop
    raw_filter_time_data = raw_data.loc[(raw_data['time'] >= time_start) & (raw_data['time'] <= time_stop)]

    plt.scatter(raw_filter_time_data["d"], raw_filter_time_data["rssi"], s=10, c="r")

    plt.scatter(b["d"], b["rssi"], s=30, c="g")
    plt.scatter([1, 3,  6,  9], ms, marker="p", s=150)
    std1 = [ms[i]+ss[i] for i in range(len(ms))]
    std2 = [ms[i]-ss[i] for i in range(len(ms))]
    plt.scatter([1, 3,  6,  9], std1, marker="*", s=150)
    plt.scatter([1, 3,  6,  9], std2, marker="*", s=150)
    plt.xlabel("distance")
    plt.ylabel("rssi")
    plt.title(f"{n} sigma for 60 seconds")
    plt.show()
