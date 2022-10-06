import matplotlib.pyplot as plt


def draw_xtime_yrssi(df_lst, row):
    if row == 1:
        ax = 1
    if row == 2:
        ax = 5
    for df in df_lst:
        plt.subplot(2, 4, ax)
        ax += 1
        plt.plot([j for j in range(df.shape[0])], df["rssi"])
        plt.xlabel("time")
        plt.ylabel("rssi")
