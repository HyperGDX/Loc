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


# def cal_inter2(circles):

#     return inter_lst


# def norm_centroid(inter_lst):

#     return loc

# def cal_inter(rssi_lst, beacon_lst):
#     # 逐渐增加幅度计算交集
#     inter = None
#     up_rate = 0.95
#     while not inter:
#         up_rate += 0.05
#         circles_parse = top_n.gen_circles_parse(rssi_lst, beacon_lst, up_rate)
#         circles_geo = gen_circles_geo(circles_parse)
#         inter = circles_geo[0]
#         for i in range(1, len(circles_geo)):
#             inter = inter.intersection(circles_geo[i])
#         if inter:
#             print(circles_parse)
#             draw_circles(circles_geo)
#             print(up_rate)
#     return inter
if __name__ == "__main__":
    xy = cal_inter(gen_circle_geo(beacon_loc, test_rssi2d_gauss))
    inter_xy = list(xy.exterior.coords)
    x = [inter_xy[i][0] for i in range(len(inter_xy))]
    y = [inter_xy[i][1] for i in range(len(inter_xy))]
    print(xy.centroid)
    print(list(xy.exterior.coords))

    # plt.plot(x, y)

    # plt.xlim(0, 5)
    # plt.ylim(0, 5)
    # plt.show()
