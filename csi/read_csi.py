
import matplotlib.pyplot as plt
from wifilib import *

path = "csi/widar_data/csi/user1_1/user1-1-1-1-1-r6.dat"

# path = r"csi\widar_data\run_lh_1.dat"
# 'id': user's id; 'a': gesture type, 'b': torso location, 'c': face orientation, 'd': repetitionnumber, 'Rx': Wi-Fi receiver id.
gesture_type_dct = {}


def get_csi_data(path):
    bf = read_bf_file(path)
    csi_list = list(map(get_scale_csi, bf))
    csi_np = (np.array(csi_list))
    csi_amp = np.abs(csi_np)

    # print("csi shape: ", csi_np.shape)
    return csi_amp
    # fig = plt.figure()
    # plt.plot(csi_amp[:, 0, 0, 1])
    # plt.show()


if __name__ == "__main__":
    for i in range(3):
        a = get_csi_data(f"csi/widar_data/csi/user1_1/user1-1-1-1-{i+1}-r6.dat")

        plt.plot(a[500:600, 0, 1, 1])
    plt.show()
