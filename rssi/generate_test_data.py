import cal_func
import random
import pandas as pd
beacon_loc = [(0, 0)]
test_loc = [(0, 1), (0, 3), (0, 6), (0, 9)]

test_data = []
# for t in range(60):
#     for i in test_loc:
#         cur_d = cal_func.cal_2pos_dist(i, beacon_loc[0])
#         cur_rssi = cal_func.d2rssi(cur_d)
#         cur_data = [t, round(cur_d, 4), round(cur_rssi+random.gauss(0, 5), 4)]
#         test_data.append(cur_data)
# test_df = pd.DataFrame(columns=["time", "d", "rssi"], data=test_data)
# test_df.to_csv("new_rssi/data/rssi_d/test.csv", index=False)

for t in range(60):
    for i in test_loc:
        if (t >= 10) & (t <= 20):
            cur_d = cal_func.cal_2pos_dist(i, beacon_loc[0])
            cur_rssi = cal_func.d2rssi(cur_d, a=random.gauss(50, 3), n=random.gauss(45, 3))
            cur_data = [t, round(cur_d, 4), round(cur_rssi, 4)]
            test_data.append(cur_data)
        else:
            cur_d = cal_func.cal_2pos_dist(i, beacon_loc[0])
            cur_rssi = cal_func.d2rssi(cur_d, a=random.gauss(45, 1), n=random.gauss(40, 1))
            cur_data = [t, round(cur_d, 4), round(cur_rssi, 4)]
            test_data.append(cur_data)

test_df = pd.DataFrame(columns=["time", "d", "rssi"], data=test_data)
test_df.to_csv("new_rssi/data/rssi_d/test.csv", index=False)
