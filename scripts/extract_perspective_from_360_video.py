#全方位画像から透視投影画像を出力するコード
import cv2
import numpy as np
import os
import tqdm

class ExtractFrom360degreeVideo:
    def __init__(self, video_path, output_folder, fov_x_deg=90, fov_y_deg=60):
        self.fov_x_deg = fov_x_deg  # 視野の幅の角度
        self.fov_y_deg = fov_y_deg  # 視野の高さの角度

        self.video_path = video_path
        self.video_filename = os.path.splitext(os.path.basename(video_path))[0]
        self.output_folder = output_folder
        self.cap = cv2.VideoCapture(video_path)
        fps= self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Video FPS: {fps}")

        if not self.cap.isOpened():
            raise Exception("Could not open video file.")

    def extract_frames(self, frame_interval=30):
        cnt = 0
        pbar = tqdm.tqdm(total=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / frame_interval), desc="Extracting frames")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print('Frame is not found.')
                break

            if cnt % frame_interval != 0:  # Extract every 'frame_interval' frames
                cnt += 1
                continue
            
            for deg_phi in range(0, 360, 15):
                for deg_theta in [90]:
                    output_image = run_equirectangular2perspective(frame, deg_phi=deg_phi, deg_theta=deg_theta, fov_x_deg=self.fov_x_deg, fov_y_deg=self.fov_y_deg)
                    output_path = os.path.join(self.output_folder, self.video_filename + f'_{cnt:05d}_{deg_phi:03d}_{deg_theta:03d}.jpg')
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    cv2.imwrite(output_path, output_image)
            cnt += 1
            pbar.update(1)
        pbar.close()

def run_equirectangular2perspective(input_image, deg_phi=0, deg_theta=0, fov_x_deg=90, fov_y_deg=60):
    height,width,ch = input_image.shape

    #透視投影画像用の平面の大きさ
    l = 2 * np.tan(np.radians(fov_x_deg / 2))  # 視野の幅
    m = 2 * np.tan(np.radians(fov_y_deg / 2))  # 視野の高さ
    rad_theta = np.pi * deg_theta / 180.0 # ラジアンに変換
    rad_phi = np.pi * deg_phi / 180.0 # ラジアンに変換


    #p_width = int(l * (width * (53.0 / 360.0)))
    #q_height = int(m * (height * (53.0 / 180.0)))
    p_width = int(np.arctan(l/2) * width / np.pi )
    #q_height = int(2*np.arctan(m/2) * height / np.pi )
    q_height = int(p_width*m/l)
    output_image = np.zeros((q_height, p_width, 3), dtype=np.uint8)

    #透視投影画像に変換
    x = np.arange(0, p_width, 1)
    y = np.arange(0, q_height, 1)
    pp, qq = np.meshgrid(x, y)

    s = (-l / (p_width - 1)) * pp + (l / 2.0)
    t = (-m / (q_height - 1)) * qq + (m / 2.0)
    x = np.sin(rad_theta) * np.cos(rad_phi) + s * np.sin(rad_phi) + t * (-(np.cos(rad_phi) * np.cos(rad_theta)))
    y = np.sin(rad_theta) * np.sin(rad_phi) + s * -(np.cos(rad_phi)) + t * -np.sin(rad_phi) * np.cos(rad_theta)
    z = np.cos(rad_theta) + s * 0 + t * np.sin(rad_theta)
    theta = np.arccos(z / np.sqrt(x * x + y * y + z * z))
    phi = np.arctan2(y, x)

    phi[phi<0] += 2*np.pi

    #if phi < 0:
    #    phi = 2 * pi + phi

    # バイリニア補間
    ud = (width * phi) / (2 * np.pi)
    vd = (height * theta) / np.pi
    x_int = ud.astype(np.int32)
    y_int = vd.astype(np.int32)
    dx, dy = ud - x_int, vd - y_int
    dx3 = np.stack([dx,dx,dx]).transpose(1,2,0)
    dy3 = np.stack([dy,dy,dy]).transpose(1,2,0)

    f00 = input_image[y_int, x_int, :]
    f01 = input_image[y_int, (x_int + 1)%width, :]
    f10 = input_image[(y_int + 1)%height, x_int, :]
    f11 = input_image[(y_int + 1)%height, (x_int + 1)%width, :]

    output_image = (1 - dy3) * ((1 - dx3) * f00 + dx3 * f01) + dy3 * ((1 - dx3) * f10 + dx3 * f11)
    output_image = output_image.astype(np.uint8)

    return output_image

def main_movie():
    video_name = r"C:\oit\yolo\main\VID_20230915_111234_00_004.mp4"
    out_folder_path = r"C:\oit\yolo\main\prediction"

    ExtractFrom360degreeVideo(video_name, out_folder_path).extract_frames(frame_interval=30*15)

if __name__ == "__main__":
    main_movie()
