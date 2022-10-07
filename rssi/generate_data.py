import tool_func
import numpy as np
import random

beacon_loc = [(0, 0), (0, 4), (4, 0), (4, 4)]
test_loc = (1.25, 2.5)

test_d = []
test_rssi = []
test_rssi_gauss = []
for i in beacon_loc:
    cur_d = tool_func.cal_2pos_dist(i, test_loc)
    cur_rssi = tool_func.d2rssi(cur_d)
    test_d.append(round(cur_d, 4))
    test_rssi.append(round(cur_rssi, 4))
    test_rssi_gauss.append(round(cur_rssi+random.gauss(0, 5), 4))
print(test_d)
print(test_rssi)
print(test_rssi_gauss)
