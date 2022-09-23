from statistics import mean
import numpy as np
import tool_func
import shapely
from shapely import geometry
import matplotlib.pyplot as plt

beacon_loc = [(0, 0), (0, 4), (4, 0), (4, 4)]
test_loc = (1.25, 2.5)

test_d = [2.7951, 1.9526, 3.7165, 3.1325]
test_rssi = [62.8558, 56.6242, 67.8054, 64.8356]
test_rssi_gauss = [60.7183, 64.797, 67.2363, 67.0921]
test_rssi2d_gauss = [2.4715, 3.1255, 3.5967, 3.567]
# for i in test_rssi_gauss:
#     test_rssi2d_gauss.append(round(tool_func.rssi2d(i), 4))
# print(test_rssi2d_gauss)


def gen_circle_geo(beacon_loc, d):
    circles = []
    for i in range(len(beacon_loc)):
        cur_cir = geometry.Point(beacon_loc[i]).buffer(d[i])
        circles.append(cur_cir)
    return circles


def cal_inter(circles):
    tool_func.draw_circles(circles)
    inter = circles[0]
    for c in circles:
        inter = inter.intersection(c)
    if inter:
        return inter
        # inter.exterior.coords.xy
    else:
        return None


def cal_real_ext(inter):
    inter_xy = list(inter.exterior.coords)
    # x = [inter_xy[i][0] for i in range(len(inter_xy))]
    # y = [inter_xy[i][1] for i in range(len(inter_xy))]
    inter_xy_add = inter_xy + [inter_xy[0]]
    real_ext = []
    for i in range(len(inter_xy)-1):
        deg = tool_func.cal_deg(inter_xy_add[i], inter_xy_add[i+1], inter_xy_add[i+2])
        if deg == None:
            continue
        if abs(deg) > 10:
            real_ext.append(inter_xy_add[i+1])
    return real_ext


def cal_centroid_norm(ext):
    cal_cen_xs = [ext[i][0] for i in range(len(ext))]
    cal_cen_ys = [ext[i][1] for i in range(len(ext))]
    cal_cen_xs_mean = mean(cal_cen_xs)
    cal_cen_ys_mean = mean(cal_cen_ys)
    return (cal_cen_xs_mean, cal_cen_ys_mean)


if __name__ == "__main__":
    inter = cal_inter(gen_circle_geo(beacon_loc, test_rssi2d_gauss))
    real_ext = cal_real_ext(inter)
    centroid_norm = cal_centroid_norm(real_ext)
    print(centroid_norm)
    print(inter.centroid)
