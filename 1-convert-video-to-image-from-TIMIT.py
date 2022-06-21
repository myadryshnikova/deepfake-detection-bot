import json
import os
import cv2
import math

base_path = './TIMIT_dataset/'
train_path = f'./train_photos/fake/'

files = os.listdir(base_path)


def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only


for filename in files:
    print(filename)
    if filename.endswith(".avi"):
        count = 0
        video_file = os.path.join(base_path, filename)
        cap = cv2.VideoCapture(video_file)
        frame_rate = cap.get(5)  # frame rate
        while cap.isOpened():
            frame_id = cap.get(1)  # current frame number
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % math.floor(frame_rate) == 0:
                if frame.shape[1] < 300:
                    scale_ratio = 2
                elif frame.shape[1] > 1900:
                    scale_ratio = 0.33
                elif 1000 < frame.shape[1] <= 1900:
                    scale_ratio = 0.5
                else:
                    scale_ratio = 1

                width = int(frame.shape[1] * scale_ratio)
                height = int(frame.shape[0] * scale_ratio)
                dim = (width, height)
                new_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

                new_filename = '{}-{:03d}.png'.format(os.path.join(train_path, get_filename_only(filename)), count)
                count = count + 1
                cv2.imwrite(new_filename, new_frame)
        cap.release()
    else:
        continue
