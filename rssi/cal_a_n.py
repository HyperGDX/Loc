from n_sigma import n_sigma

import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt


def d2r_func(x, a, b):
    return a + b * np.log10(x)


def cal_by_opt(d_r: pd.DataFrame):
    x0, y0 = d_r["d"], d_r["rssi"]
    a4, b4 = optimize.curve_fit(d2r_func, x0, y0)[0]
    return round(a4, 4), round(b4, 4)


if __name__ == "__main__":
    raw_data = pd.read_csv("rssi/data/rssi_d/rssi_d.csv")
    for i in range(10, 51, 5):
        sigma_data = n_sigma(raw_data, time_start=0, time_stop=60, n=i/10, direct_use=True)
        a, b = cal_by_opt(sigma_data)
        print(a, b)
