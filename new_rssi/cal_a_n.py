import numpy as np
import pandas as pd
from scipy import optimize


def d2r_func(x, a, b):
    return a + b * np.log10(x)


def cal_by_opt(d_r: pd.DataFrame):
    x0, y0 = d_r["d"], d_r["rssi"]
    a4, b4 = optimize.curve_fit(d2r_func, x0, y0)[0]
    return round(a4, 4), round(b4, 4)
