import matplotlib.pyplot as plt
import pandas as pd

import cal_a_n
import draw_func
import filter
import n_sigma


def time_slot_proc(df: pd.DataFrame, time_begin, time_stop):
    first_time = df["time"][0]
    new_df = df.loc[(df["time"] >= first_time+time_begin) & (df["time"] <= first_time+time_begin+time_stop)]
    return new_df.reset_index(drop=True)


def dif_d_proc(df: pd.DataFrame):
    df_lst = [df.loc[df["d"] == i] for i in [1, 3, 6, 9]]
    return df_lst


def concat_df_lst(df_lst):
    new_df = pd.DataFrame(columns=["time", "d", "rssi"])
    for df in df_lst:
        new_df = pd.concat([new_df, df], axis=0, ignore_index=True)
    return new_df


if __name__ == "__main__":
    ### config ###
    time_begin = 0
    time_stop = 60
    sigma = 1.5
    win = 7
    ### read raw data ###
    # raw_df = pd.read_csv("new_rssi/data/rssi_d/rssi_d.csv")
    raw_df = pd.read_csv("new_rssi/data/rssi_d/test.csv")
    # raw_df = pd.read_csv(r"new_rssi\data\rssi_d\rssi_d.csv")

    ### time slot ###
    time_slot_df = time_slot_proc(raw_df, time_begin, time_stop)
    time_slot_df_lst = dif_d_proc(time_slot_df)
    draw_func.draw_xtime_yrssi(time_slot_df_lst, row=1)

    ### optim ###
    a, n = cal_a_n.cal_by_opt(raw_df)
    print("raw data  a:", a, "n", n)

    ### n sigma ###
    n_sigma_df_lst = n_sigma.n_sigma_df_lst(time_slot_df_lst, sigma)
    draw_func.draw_xtime_yrssi(n_sigma_df_lst, row=2)

    ### concat ###
    concat_df = concat_df_lst(n_sigma_df_lst)
    ### optim ###
    a, n = cal_a_n.cal_by_opt(concat_df)
    print("n sigma  a:", a, "n", n)

    # MeanFilter
    mean_filter_df_lst = filter.MeanFilterDFlst(n_sigma_df_lst, win)
    draw_func.draw_xtime_yrssi(mean_filter_df_lst, row=2)
    ### concat ###
    concat_df = concat_df_lst(mean_filter_df_lst)
    ### optim ###
    a, n = cal_a_n.cal_by_opt(concat_df)
    print("MeanFilter a:", a, "n", n)

    ### KalmanFilter ###
    kal_filter_df_lst = filter.KalmanFilterDFlst(mean_filter_df_lst)
    draw_func.draw_xtime_yrssi(kal_filter_df_lst, row=2)

    ### concat ###
    concat_df = concat_df_lst(kal_filter_df_lst)
    ### optim ###
    a, n = cal_a_n.cal_by_opt(concat_df)
    print("KalmanFilter a:", a, "n", n)
    plt.show()
