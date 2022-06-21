import json
import os
from distutils.dir_util import copy_tree
import shutil
import numpy as np
import splitfolders

base_path = './train_sample_videos/'
dataset_path = './prepared_dataset/'

os.makedirs(dataset_path, exist_ok=True)

tmp_fake_path = './tmp_fake_faces'

os.makedirs(tmp_fake_path, exist_ok=True)


def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only


with open(os.path.join(base_path, 'metadata.json')) as metadata_json:
    metadata = json.load(metadata_json)
    print(len(metadata))

real_path = os.path.join(dataset_path, 'real')

os.makedirs(real_path, exist_ok=True)

fake_path = os.path.join(dataset_path, 'fake')

os.makedirs(fake_path, exist_ok=True)

for filename in metadata.keys():
    tmp_path = os.path.join(os.path.join(base_path, get_filename_only(filename)), 'faces')

    if os.path.exists(tmp_path):
        if metadata[filename]['label'] == 'REAL':
            copy_tree(tmp_path, real_path)
        elif metadata[filename]['label'] == 'FAKE':
            copy_tree(tmp_path, tmp_fake_path)

all_real_faces = [f for f in os.listdir(real_path) if os.path.isfile(os.path.join(real_path, f))]
print('Total Number of Real faces: ', len(all_real_faces))

all_fake_faces = [f for f in os.listdir(tmp_fake_path) if os.path.isfile(os.path.join(tmp_fake_path, f))]
print('Total Number of Fake faces: ', len(all_fake_faces))

random_faces = np.random.choice(all_fake_faces, len(all_real_faces), replace=False)
for fname in random_faces:
    src = os.path.join(tmp_fake_path, fname)
    dst = os.path.join(fake_path, fname)
    shutil.copyfile(src, dst)

splitfolders.ratio(dataset_path, output='split_dataset', seed=1377, ratio=(.8, .1, .1))