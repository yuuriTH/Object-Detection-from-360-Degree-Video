import pandas as pd

data = pd.read_csv("./result.csv")

print(data["iou"])
