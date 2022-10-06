import matplotlib.pyplot as plt
import pandas as pd

import cal_a_n
import filter
import n_sigma
import tool_func


def sigma_filter_proc(df: pd.DataFrame, time_start, time_stop, n, filter_kind):
    # draw raw data with time slot
    first_time = df.loc[0][0]
    time_df = df.loc[(df['time'] >= first_time+time_start) & (df['time'] <= first_time+time_start+time_stop)]
    time_df_lst = [time_df.loc[time_df["d"]==i] for i in [1,3,6,9]]
    tool_func.draw_xtime_yrssi(time_df_lst)

    # do sigma
    sigma_data_lst = n_sigma.n_sigma(df, time_start, time_stop, n=n, direct_use=False)
    tool_func.draw_xtime_yrssi(sigma_data_lst)
    # do filter
    if filter_kind == "Mean":
        filter_data_lst = filter.MeanFilterDF(sigma_data_lst, win=3)
    tool_func.draw_xtime_yrssi(filter_data_lst)
    plt.show()
    # draw d & rssi after sigma and filter
    for f_d in filter_data_lst:
        plt.scatter(f_d["d"],f_d["rssi"])
    plt.show()



    return filter_data_lst


if __name__ == "__main__":
    raw_data = pd.read_csv("rssi/data/rssi_d/rssi_d.csv")
    # for i in range(10, 51, 5):
    #     sigma_data = n_sigma.n_sigma(raw_data, time_start=0, time_stop=60, n=i/10, direct_use=True)
    #     a, b = cal_a_n.cal_by_opt(sigma_data)
    #     print(a, b)
    a = sigma_filter_proc(raw_data, time_start=0, time_stop=60, n=1.5, filter_kind="Mean")
    print(a)
    