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


def cal_deg(p, q, r):
    x1 = p[0]
    y1 = p[1]
    x2 = q[0]
    y2 = q[1]
    x3 = r[0]
    y3 = r[1]
    try:
        k1 = (y2-y1)/(x2-x1)
        k2 = (y3-y2)/(x3-x2)
        deg1 = math.degrees(math.atan(k1))
        deg2 = math.degrees(math.atan(k2))
        return deg2-deg1
    except ZeroDivisionError:
        return None


if __name__ == "__main__":
    rssi = 90
    d = rssi2d(rssi)
    print(d)
