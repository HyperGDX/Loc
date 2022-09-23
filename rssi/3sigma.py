import numpy as np
import random

test_d = [2.7951, 1.9526, 3.7165, 3.1325]
test_rssi = [62.8558, 56.6242, 67.8054, 64.8356]
test_rssi_gauss = [60.7183, 64.797, 67.2363, 67.0921]
# test_rssi_gauss = []
# for r in test_rssi:
#     test_rssi_gauss.append(round(r+random.gauss(0, 5), 4))
test_rssi_nd = np.array(test_rssi)

# raw
miu = np.mean(test_rssi_nd)
sigma = np.std(test_rssi_nd)
print(miu, sigma)

# gauss
test_rssi_gauss_nd = np.array(test_rssi_gauss)
miu = np.mean(test_rssi_gauss_nd)
sigma = np.std(test_rssi_gauss_nd)
print(miu, sigma)
