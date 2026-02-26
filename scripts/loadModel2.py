from ultralytics import YOLO
import pandas as pd

model = YOLO("yolo11n.pt")

results = model.val(data="./data.yaml", plots=True)
print(results.confusion_matrix.to_df())
df = results.confusion_matrix.to_df() 
print(type(df))
print(df)
