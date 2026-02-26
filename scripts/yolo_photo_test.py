from ultralytics import YOLO
import cv2
import os

def detect_and_save(photo_path, save_img_path, save_label_path):
    # 学習済みモデルのパス
    model = YOLO("runs/detect/train38/weights/best.pt")

    # 画像のパス
    #photo_path = r"C:\oit\yolo\main\video4_photo\VID_20230915_111234_00_004_02700_315_090.jpg"

    # 出力先
    #save_img_path = "output.jpg"
    #save_label_path = "output.txt"

    # 画像を読み込み
    #print(photo_path)
    img = cv2.imread(str(photo_path))
    h, w = img.shape[:2]

    # YOLOで物体検出
    results = model(img, verbose=False)
    result = results[0]  # 1画像なので1つだけ

    # ------------------------------
    # ① YOLO形式のtxtを書き出す
    # ------------------------------
    with open(save_label_path, "w") as f:
        for box in result.boxes:
            cls = int(box.cls[0])  # クラスID
            conf = float(box.conf[0])  # 確信度(必要なら)

            # xyxy形式の座標を取得
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # YOLO形式に変換 (正規化)
            x_center = (x1 + x2) / 2.0 / w
            y_center = (y1 + y2) / 2.0 / h
            bbox_width = (x2 - x1) / w
            bbox_height = (y2 - y1) / h

            # txt形式: class x_center y_center width height
            f.write(f"{cls} {x_center} {y_center} {bbox_width} {bbox_height}\n")

    #print("bbox座標を output.txt に保存しました。")

    # ------------------------------
    # ② 結果画像を表示＋保存
    # ------------------------------
    annotated_frame = result.plot()

    # cv2.imshow("YOLO Detection", annotated_frame)
    cv2.imwrite(save_img_path, annotated_frame)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return annotated_frame

if __name__ == "__main__":
    # 入力画像（テスト用）
    photo_path = r"C:\oit\yolo\main\video4_photo\VID_20230915_111234_00_004_02700_315_090.jpg"
    #photo_path = r"C:\oit\yolo\main\prediction"

    # 出力先
    save_img_path = "output.jpg"
    save_label_path = "output.txt"

    print("YOLO検出を実行します...")
    detect_and_save(photo_path, save_img_path, save_label_path)
    print("完了！")
