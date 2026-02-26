import os
import pandas as pd

label_dir = r"C:/oit/yolo/main/test"

# 元画像サイズ
IMG_W = 959
IMG_H = 553
IMG_AREA = IMG_W * IMG_H

# 出力CSV
output_csv = r"C:/oit/yolo/main/bbox_max_area_percent_only.csv"

rows = []

#txtファイル1つずつ処理
for file in os.listdir(label_dir):
    if not file.endswith(".txt"):
        continue

    path = os.path.join(label_dir, file)

    with open(path, "r") as f:
        lines = f.readlines()

    #txt空
    if len(lines) == 0:
        rows.append({
            "image": file.replace(".txt", ".jpg"),
            "max_area_percent": 0.0
        })
        continue

    areas_ratio = []

    #各バウンディングボックス
    for line in lines:
        parts = line.strip().split()

        # 5列（class xc yc w h）または 6列（+ conf）対応
        if len(parts) < 5:
            continue

        _, xc, yc, w, h = map(float, parts[:5])

        area_ratio = w * h  # 画像全体に対する割合
        areas_ratio.append(area_ratio)

    # bboxが1つも有効でなかった場合
    if len(areas_ratio) == 0:
        rows.append({
            "image": file.replace(".txt", ".jpg"),
            "max_area_percent": 0.0
        })
        continue

    #最大
    max_area_percent = max(areas_ratio) * 100

    rows.append({
        "image": file.replace(".txt", ".jpg"),
        "max_area_percent": max_area_percent
    })

df = pd.DataFrame(rows)

#0%除外
df_filtered = df[df["max_area_percent"] > 0]

#大きい順
df_filtered = df_filtered.sort_values(
    "max_area_percent",
    ascending=False
)

#小数点
df_filtered["max_area_percent"] = df_filtered["max_area_percent"].round(3)

#CSV保存
df_filtered.to_csv(
    output_csv,
    index=False,
    encoding="utf-8-sig"
)

print("CSVを保存しました:", output_csv)
print(df_filtered.head())
