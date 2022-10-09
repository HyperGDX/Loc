import pandas as pd


raw_data = pd.read_csv(r"rssi/data/UJIndoorLoc/trainingData.csv")
print(raw_data.iloc[0, 522:525])
print(raw_data.shape[0])
