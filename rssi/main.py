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
    win = 5
    ### read raw data ###
    # raw_df = pd.read_csv("new_rssi/data/rssi_d/rssi_d.csv")
    raw_df = pd.read_csv("rssi/data/rssi_d/test.csv")
    # raw_df = pd.read_csv(r"new_rssi\data\rssi_d\rssi_d.csv")

    ### time slot ###
    time_slot_df = time_slot_proc(raw_df, time_begin, time_stop)
    time_slot_df_lst = dif_d_proc(time_slot_df)
    draw_func.draw_xtime_yrssi(time_slot_df_lst, draw_row=1)
    # optim #
    raw_a, raw_n = cal_a_n.cal_by_opt(time_slot_df)
    print("raw data  a:", raw_a, "n", raw_n)

    ### n sigma ###
    n_sigma_df_lst = n_sigma.n_sigma_df_lst(time_slot_df_lst, sigma)
    draw_func.draw_xtime_yrssi(n_sigma_df_lst, draw_row=2)
    # concat #
    n_sigma_df = concat_df_lst(n_sigma_df_lst)
    # optim #
    sigma_a, sigma_n = cal_a_n.cal_by_opt(n_sigma_df)
    print("n sigma  a:", sigma_a, "n", sigma_n)

    # MeanFilter
    mean_filter_df_lst = filter.MeanFilterDFlst(n_sigma_df_lst, win)
    draw_func.draw_xtime_yrssi(mean_filter_df_lst, draw_row=2)
    # concat #
    mean_filter_df = concat_df_lst(mean_filter_df_lst)
    # optim #
    mean_a, mean_n = cal_a_n.cal_by_opt(mean_filter_df)
    print("MeanFilter a:", mean_a, "n", mean_n)

    ### KalmanFilter ###
    kal_filter_df_lst = filter.KalmanFilterDFlst(mean_filter_df_lst)
    draw_func.draw_xtime_yrssi(kal_filter_df_lst, draw_row=2)
    # concat #
    kal_filter_df = concat_df_lst(kal_filter_df_lst)
    # optim #
    kal_a, kal_n = cal_a_n.cal_by_opt(kal_filter_df)
    print("KalmanFilter a:", kal_a, "n", kal_n)
    plt.show()

    # draw xd yrssi 4 subplot
    all_df = [time_slot_df, n_sigma_df, mean_filter_df, kal_filter_df]
    a_lst = [raw_a, sigma_a, mean_a, kal_a]
    n_lst = [raw_n, sigma_n, mean_n, kal_n]
    for i in range(len(a_lst)):
        plt.subplot(1, 4, i+1)
        draw_func.draw_xd_yrssi(all_df[i], a_lst[i], n_lst[i])
    plt.show()
