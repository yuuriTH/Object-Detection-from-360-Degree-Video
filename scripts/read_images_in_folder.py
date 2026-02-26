import cv2
import os
from pathlib import Path
from yolo_photo_test import detect_and_save
# https://docs.python.org/ja/3/library/pathlib.html
# pathlib https://note.nkmk.me/python-pathlib-usage/
# pathlib glob https://note.nkmk.me/python-pathlib-iterdir-glob/

def main():
    img_list = list(Path(r"C:\oit\yolo\main\prediction").glob("*.jpg")) # **/*  -->  **/*.jpg, **/*.bmp
    #print(img_list)
    img_path = r"C:\oit\yolo\main\prediction"

    imgs = []
    for img_path in img_list:
        imgs.append(detect_and_save(img_path, os.path.basename(img_path), str(os.path.splitext(os.path.basename(img_path))[0])+ '.txt'))
    # imgs = [cv2.imread(str(img_path)) for img_path in img_list]

    #for img_path, img in zip(img_list, imgs):
    #    cv2.imshow(str(img_path), img)
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
