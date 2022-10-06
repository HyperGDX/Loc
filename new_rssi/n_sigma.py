import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def n_sigma_df(df: pd.DataFrame, n):
    cur_rssi = df["rssi"]
    rule = ((cur_rssi.mean()-n*cur_rssi.std()) < cur_rssi) & ((cur_rssi.mean()+n*cur_rssi.std()) > cur_rssi)
    after_df = df.loc[rule]
    return after_df.reset_index(drop=True)


def n_sigma_df_lst(df_lst, n):
    new_df_lst = []
    for df in df_lst:
        new_df_lst.append(n_sigma_df(df, n))
    return new_df_lst


if __name__ == "__main__":
    raw_data = pd.read_csv("rssi/data/rssi_d/rssi_d.csv")
