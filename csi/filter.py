import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def MeanFilter1D(data: pd.DataFrame, win):
    # ans_df = pd.DataFrame(columns=["time", "d", "rssi"])
    d_l = data.shape[0]
    for i in range(d_l):
        sum = 0.0
        for j in range(i-win//2, i+win//2+1):
            if j >= d_l or j < 0:
                sum += data.iloc[i]
            else:
                sum += data.iloc[j]
        data.iloc[i] = sum/win
    return data.reset_index(drop=True)


def MeanFilterDFlst(df_lst, win=5):
    ans_lst = []
    for df in df_lst:
        ans_lst.append(MeanFilter1D(df, win))
    return ans_lst


def MedianFilter1D(data, win=3):
    ans = []
    d_l = len(data)
    for i in range(d_l):
        tmp = []
        for j in range(i-win//2, i+win//2+1):
            if j >= d_l or j < 0:
                tmp.append(data[i])
            else:
                tmp.append(data[j])
        ans.append(sorted(tmp)[win//2])
    return ans


def GaussFilter(data):
    pass


def ABFilter1D(data, alpha=0.2):
    ans = [data[0]]
    for i in range(1, len(data)):
        ans.append(data[i]*(1-alpha)+data[i-1]*alpha)
    return ans


def KalmanFilter1DF(df: pd.DataFrame):
    new_df = df.copy(deep=True)
    # 滤波效果主要调整参数：
    # 过程噪声方差q(越小越相信预测，反之亦然)， 观测噪声方差r(越小越相信观测，反之亦然)
    q, r = 0.1, 2
    # 状态均值x， 过程噪声均值w，方差p
    x, w, p = df["rssi"][0], 0, 0

    def kalman_filter(z):
        # 预测
        nonlocal x, p
        x_ = x + w
        p_ = p + q
        k = p_ / (p_ + r)
        # 更新
        x = x_ + k * (z - x_)
        p = (1-k) * p_
        return x

    for i in range(df.shape[0]):
        new_df["rssi"][i] = kalman_filter(df["rssi"][i])
    return new_df


def KalmanFilter1ND(nd: np.ndarray):
    new_nd = nd.copy(deep=True)
    # 滤波效果主要调整参数：
    # 过程噪声方差q(越小越相信预测，反之亦然)， 观测噪声方差r(越小越相信观测，反之亦然)
    q, r = 0.1, 2
    # 状态均值x， 过程噪声均值w，方差p
    x, w, p = nd[0], 0, 0

    def kalman_filter(z):
        # 预测
        nonlocal x, p
        x_ = x + w
        p_ = p + q
        k = p_ / (p_ + r)
        # 更新
        x = x_ + k * (z - x_)
        p = (1-k) * p_
        return x

    for i in range(len(nd)):
        new_nd[i] = kalman_filter(nd[i])
    return new_nd


def KalmanFilterDFlst(df_lst):
    new_df_lst = []
    for df in df_lst:
        new_df_lst.append(KalmanFilter1DF(df))
    return new_df_lst


# if __name__ == "__main__":
#     data = [-77, -95, -82, -80, -80, -84, -83, -78, -63, -78, -77, -86, -62, -83,
#             -79, -68, -79, -78, -83, -82, -73, -79, -73, -59, -78, -59, -72, -72,
#             -77, -84, -77, -72, -60, -80, -68, -79, -79, -66, -67, -79]
#     mean = MeanFilter1D(data, 5)
#     med = MedianFilter1D(data, 3)
#     ab = ABFilter1D(data, 0.2)
#     kal = KalmanFilter1(data)

#     plt.plot(data, "k", linewidth=2)
#     plt.plot(mean, "r")
#     plt.plot(med, "g")
#     plt.plot(ab, "b")
#     plt.plot(kal, "c")
#     plt.show()
