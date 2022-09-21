import math


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
