import os.path

import cv2
import tensorflow as tf
from mtcnn import MTCNN

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

physical_devices = tf.config.list_physical_devices('GPU')


def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only


def prepare_dataset_facebook(label):
    train_path = f'./split_dataset/train/{label}'
    path = f'./train_photos/{label}'
    frame_images = [x for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))]

    for frame in frame_images:
        print(f'Work with {frame}')
        detector = MTCNN()
        image = cv2.cvtColor(cv2.imread(os.path.join(path, frame)), cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(image)
        count = 0

        for result in results:
            bounding_box = result['box']
            confidence = result['confidence']
            if len(results) < 2 or confidence > 0.95:
                margin_x = bounding_box[2] * 0.3
                margin_y = bounding_box[3] * 0.3
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
                new_filename = '{}-{:02d}.png'.format(os.path.join(train_path, get_filename_only(frame)), count)
                count = count + 1
                cv2.imwrite(new_filename, cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR))


prepare_dataset_facebook('fake')
prepare_dataset_facebook('real')
