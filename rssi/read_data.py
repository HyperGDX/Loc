import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt

# 1 3 6 9
# 11 12 13 14
idx_d_map = {
    11: 1,
    12: 3,
    13: 6,
    14: 9
}
raw_df = pd.read_csv(r"rssi\data\5.11\data\newdata.CSV")
t = []
d = []
r = []
for i in range(len(raw_df)):
    cur_t, cur_idx, _, cur_r = raw_df.iloc[i]
    if cur_idx in idx_d_map:
        t.append(cur_t)
        d.append(int(idx_d_map[cur_idx]))
        r.append(-1 * int(cur_r))
d_r_nd = np.array([d, r], dtype=np.float32)
# 3*3067


def d2r_func(x, a, b):
    return a + b * np.log10(x)


def cal_a_n(d_r):
    x0, y0 = d_r[0], d_r[1]
    a4, b4 = optimize.curve_fit(d2r_func, x0, y0)[0]
    return round(a4, 4), round(b4, 4)


def draw_curve_img(d_r):
    x, y = d_r[0], d_r[1]
    plt.scatter(x, y)


def draw_cal_a_n_img(a, b):
    x = [i for i in range(1, 10)]
    y = [d2r_func(i, a, b) for i in x]
    plt.plot(x, y)


if __name__ == "__main__":
    print(d_r_nd.shape)
    res = cal_a_n(d_r_nd[:, :])
    print(res)
    draw_cal_a_n_img(res[0], res[1])
    draw_curve_img(d_r_nd[:, :])
    plt.show()
