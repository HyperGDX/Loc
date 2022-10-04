import pandas as pd
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

raw_data = pd.read_csv("rssi/data/rssi_d/rssi_d.csv")


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
