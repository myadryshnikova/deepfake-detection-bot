import json
import os.path

import cv2
import tensorflow as tf
from mtcnn import MTCNN

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

physical_devices = tf.config.list_physical_devices('GPU')

base_path = './train_sample_videos/'


def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only


with open(os.path.join(base_path, 'metadata.json')) as metadata_json:
    metadata = json.load(metadata_json)

for filename in metadata.keys():
    tmp_path = os.path.join(base_path, get_filename_only(filename))
    frame_images = [x for x in os.listdir(tmp_path) if os.path.isfile(os.path.join(tmp_path, x))]
    faces_path = os.path.join(tmp_path, 'faces')
    os.makedirs(faces_path, exist_ok=True)

    for frame in frame_images:
        detector = MTCNN()
        image = cv2.cvtColor(cv2.imread(os.path.join(tmp_path, frame)), cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(image)
        count = 0

        for result in results:
            bounding_box = result['box']
            confidence = result['confidence']
            if len(results) < 2 or confidence > 0.95:
                margin_x = bounding_box[2] * 0.3  # 30% as the margin
                margin_y = bounding_box[3] * 0.3  # 30% as the margin
                x1 = int(bounding_box[0] - margin_x)
                if x1 < 0:
                    x1 = 0
                x2 = int(bounding_box[0] + bounding_box[2] + margin_x)
                if x2 > image.shape[1]:
                    x2 = image.shape[1]
                y1 = int(bounding_box[1] - margin_y)
                if y1 < 0:
                    y1 = 0
                y2 = int(bounding_box[1] + bounding_box[3] + margin_y)
                if y2 > image.shape[0]:
                    y2 = image.shape[0]
                crop_image = image[y1:y2, x1:x2]
                new_filename = '{}-{:02d}.png'.format(os.path.join(faces_path, get_filename_only(frame)), count)
                count = count + 1
                cv2.imwrite(new_filename, cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR))
