import numpy as np
import math


def cal_2pos_dist(pos1, pos2):
    """
    计算两点之间的距离
    :param pos1: tuple(x1,y1)
    :param pos2: tuple(x2,y2)
    :return dist: float(d)
    """
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def d2rssi(d, a=45, n=40):
    """
    将单个距离换算成rssi,输入的rssi为负值
    """
    rssi = n * np.log10(d) + a
    return rssi


def d2r_func(x, a, b):
    return a + b * np.log10(x)


def rssi2d(rssi, a=45, n=40):
    """
    将单个rssi换算成距离,输出的rssi为负值
    """
    d = 10 ** ((rssi-a) / n)
    return d


if __name__ == "__main__":
    rssi = 90
    d = rssi2d(rssi)
    print(d)
