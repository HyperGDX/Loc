import tool_func
import numpy as np

beacon_loc = [(0, 0), (0, 4), (4, 0), (4, 4)]
test_loc = (1.25, 2.5)

test_d = []
test_rssi = []
for i in beacon_loc:
    cur_d = tool_func.cal_2pos_dist(i, test_loc)
    cur_rssi = tool_func.d2rssi(cur_d)
    test_d.append(cur_d)
    test_rssi.append
print(test_d)
