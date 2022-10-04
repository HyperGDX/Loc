import numpy as np
import pandas as pd

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
    new_df = df.loc[(df['time'] >= time_start) & (df['time'] <= time_stop)]
    ans_df = pd.DataFrame(columns=["time", "d", "rssi"])
    for i in [1, 3, 6, 9]:
        cur = new_df.loc[new_df['d'] == i]
        cur_after = delete_out3sigma(cur, n)
        pd.concat([ans_df, cur_after], ignore_index=True)
    return ans_df


if __name__ == "__main__":
    raw_data = pd.read_csv("rssi/data/rssi_d/rssi_d.csv")
    print(n_sigma_df(raw_data, time_start=0, time_stop=60, n=1))
