import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


new_df = pd.DataFrame(columns=["time", "d", "rssi"], data=[["1", "2", "3"], ["4", "5", "6"]])
print(new_df)
print(new_df["rssi"][0])
