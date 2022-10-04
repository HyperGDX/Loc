# 1650976505

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

raw_data = pd.read_csv("rssi/data/rssi_d/rssi_d.csv")

a = raw_data.loc[raw_data['time']]
print(a)
