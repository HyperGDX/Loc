import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cal_func


def draw_xtime_yrssi(df_lst, draw_row):
    if draw_row == 1:
        ax = 1
    if draw_row == 2:
        ax = 5
    for df in df_lst:
        plt.subplot(2, 4, ax)
        ax += 1
        plt.plot([j for j in range(df.shape[0])], df["rssi"])
        plt.xlabel("time")
        plt.ylabel("rssi")


def draw_xd_yrssi(df: pd.DataFrame, a, n):
    for i in [1, 3, 6, 9]:
        cur_df = df.loc[df["d"] == i]
        plt.scatter(cur_df["d"], cur_df["rssi"], s=10)
        plt.xlabel("d")
        plt.ylabel("rssi")
    draw_an(a, n)


def draw_an(a, n):
    x = np.arange(1, 10, 0.5)
    y = cal_func.d2rssi(x, a, n)
    plt.plot(x, y, c="r")
    ybase = cal_func.d2rssi(x, a=45, n=40)
    plt.plot(x, ybase, c="g")


def draw_circles(circles_geo):
    for cir in circles_geo:
        x, y = cir.exterior.xy
        plt.plot(x, y)
