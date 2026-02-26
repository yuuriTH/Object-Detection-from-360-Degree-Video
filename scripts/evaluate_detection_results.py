import os
from pathlib import Path
import pandas as pd
import cv2

def calc_iou(a, b):
    # a = [x, y, w, h]
    # b = [x, y, w, h]
    _, ax1, ay1, aw, ah = a
    _, bx1, by1, bw, bh = b

    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh

    union = area_a + area_b - intersection
    return intersection / union if union != 0 else 0


# -------------------------
# テキストから数値取り出し
# -------------------------
def load_txt_values(path):
    with open(path, "r") as f:
        nums = []
        while True:
            line = f.readline().strip()
            if line == "":
                break
            nums.append(list(map(float, line.split())))
        return nums   # 例) [x, y, w, h]


# -------------------------
# メイン処理
# -------------------------

label_dir = r"C:/oit/yolo/main/test"        # テストのフォルダ
pred_dir = r"C:/oit/yolo/main/prediction"     # 予測のフォルダ
save_csv = "result.csv"

# mikensyutu = Path()

records = []   # pandas に渡す最終データ

pred_files = list(Path(pred_dir).glob('*.txt'))
#label_files = list(Path(label_dir).glob('*.txt'))
label_files = [
    p for p in Path(label_dir).glob('*.txt')
    if p.name != "classes.txt"
]
print(f"label:{len(label_files)}, pred:{len(pred_files)}")

pred_basenames_list = [os.path.basename(x) for x in pred_files]
label_basenames_list = [os.path.basename(x) for x in label_files]

TP= TN= FP= FN = 0

mikensyutsu_dir = "C:/oit/yolo/main/FN/"
gokensyutsu_dir = "C:/oit/yolo/main/FP/"

for tfile in pred_basenames_list:
    # 正解と予測でどちらもtxtファイルがあるとき
    file_name = tfile.split('.')[0]
    img = cv2.imread(pred_dir + "/" + file_name + ".jpg")
    if tfile in label_basenames_list:          # ← 同じファイルが存在したら処理
        pred_path = os.path.join(pred_dir, tfile)
        label_path = os.path.join(label_dir, tfile)

        # 値を読み込み
        pred_vals = load_txt_values(pred_path)
        label_vals = load_txt_values(label_path)

        # 正解も予測もbBoxが出現してないとき
        if len(pred_vals) == 0 and len(label_vals) == 0:
            # 真陰性
            TN = TN + 1
            continue
        # 正解にはbBoxあるけど、予測にはbBoxがないとき
        elif len(pred_vals) == 0 and len(label_vals) != 0:
            # 偽陰性（未検出）
            output_path = mikensyutsu_dir + file_name + ".jpg"
            if img is not None:
                cv2.imwrite(output_path, img)
            else:
                print("FN:" + file_name)
            FN = FN + 1
            continue
        # 正解にはbBoxないけど、予測にはbBoxがあるとき
        elif len(pred_vals) != 0 and len(label_vals) == 0:
            # 偽陽性（誤検出）
            output_path = gokensyutsu_dir + file_name + ".jpg"
            if img is not None:
                cv2.imwrite(output_path, img)
            else:
                print("FP:" + file_name)
            FP = FP + 1
            continue

        # IOU 計算
        for pred_val in pred_vals:
            for label_val in label_vals:
                iou = calc_iou(pred_val, label_val)
                # 記録用データ作成
                records.append({
                    "filename": tfile,
                    "label_vals": label_val,
                    "pred_vals": pred_val,
                    "iou": iou
                })
            
            # IoUが0.5以上のとき
            if iou >= 0.5:
                # 真陽性
                TP = TP + 1
            # IoUが0.5未満のとき
            else:
                # 偽陽性（誤検出）
                output_path = gokensyutsu_dir + file_name + ".jpg"
                if img is not None:
                    cv2.imwrite(output_path, img)
                else:
                    print("FP:" + file_name)
                FP = FP + 1
    # 予測にはtxtファイルがあるが、正解にはtxtファイルがない(正解にはbBoxが出現していない)とき
    else:
        pred_path = os.path.join(pred_dir, tfile)
        pred_vals = load_txt_values(pred_path)
        # 予測でbBoxが出現してないし、正解にもbBoxが出現してないとき
        if len(pred_vals) == 0:
            # 真陰性
            TN = TN + 1
        # 予測ではbBoxが出現してしまったが、正解にはbBoxが出現してしていないとき
        else:
            # 偽陽性（誤検出）
            output_path = gokensyutsu_dir + file_name + ".jpg"
            if img is not None:
                cv2.imwrite(output_path, img)
            else:
                print("FP:" + file_name)
            FP = FP + 1

# -------------------------
# pandas に変換 → CSV 保存
# -------------------------
df = pd.DataFrame(records)
df.to_csv(save_csv, index=False)

print("保存完了:", save_csv)
print("TP =", TP, "TN =", TN, "FP =", FP, "FN =", FN)

Accuracy = (TP + TN) / (TP + FP + TN + FN)
ErrorRate= 1 - Accuracy
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
f1 = TP * 2 / (TP * 2 + FN + FP)


print("Accuracy =", Accuracy, "ErrorRate =", ErrorRate, "Precision =", Precision, "Recall =", Recall, "f1 =", f1)
