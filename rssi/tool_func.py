import numpy as np
import math
import matplotlib.pyplot as plt


def cal_2pos_dist(pos1, pos2):
    """
    计算两点之间的距离
    :param pos1: tuple(x1,y1)
    :param pos2: tuple(x2,y2)
    :return dist: float(d)
    """
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def d2rssi(d, a=-45, n=4):
    rssi = -1 * (a - 10 * n * math.log10(d))
    return rssi


def rssi2d(rssi, a=-45, n=4):
    d = 10 ** ((a + rssi) / (10 * n))
    return d


def draw_circles(circles_geo):
    for cir in circles_geo:
        x, y = cir.exterior.xy
        plt.plot(x, y)


def insec(p1, r1, p2, r2):
    x = p1[0]
    y = p1[1]
    R = r1
    a = p2[0]
    b = p2[1]
    S = r2
    d = math.sqrt((abs(a-x))**2 + (abs(b-y))**2)
    if d > (R+S) or d < (abs(R-S)):
        print("Two circles have no intersection")
        return
    elif d == 0 and R == S:
        print("Two circles have same center!")
        return
    else:
        A = (R**2 - S**2 + d**2) / (2 * d)
        h = math.sqrt(R**2 - A**2)
        x2 = x + A * (a-x)/d
        y2 = y + A * (b-y)/d
        x3 = round(x2 - h * (b - y) / d, 2)
        y3 = round(y2 + h * (a - x) / d, 2)
        x4 = round(x2 + h * (b - y) / d, 2)
        y4 = round(y2 - h * (a - x) / d, 2)
        print(x3, y3)
        print(x4, y4)
        c1 = np.array([x3, y3])
        c2 = np.array([x4, y4])
        return c1, c2
