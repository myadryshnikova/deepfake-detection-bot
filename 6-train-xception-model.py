import os

import pandas as pd

import tensorflow as tf
import os

import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

dataset_path = './split_dataset/'

tmp_debug_path = './tmp_debug'
print('Creating Directory: ' + tmp_debug_path)
os.makedirs(tmp_debug_path, exist_ok=True)


def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only

input_size = 128
batch_size_num = 32
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
test_path = os.path.join(dataset_path, 'test')

train_datagen = ImageDataGenerator(
    rescale=1/255,    #rescale the tensor values to [0,1]
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(input_size, input_size),
    color_mode="rgb",
    class_mode="binary",  #"categorical", "binary", "sparse", "input"
    batch_size=batch_size_num,
    shuffle=True
)

val_datagen = ImageDataGenerator(
    rescale = 1/255    #rescale the tensor values to [0,1]
)

val_generator = val_datagen.flow_from_directory(
    directory=val_path,
    target_size=(input_size, input_size),
    color_mode="rgb",
    class_mode="binary",  #"categorical", "binary", "sparse", "input"
    batch_size=batch_size_num,
    shuffle=True
)

test_datagen = ImageDataGenerator(
    rescale=1/255    #rescale the tensor values to [0,1]
)

test_generator = test_datagen.flow_from_directory(
    directory=test_path,
    classes=['real', 'fake'],
    target_size=(input_size, input_size),
    color_mode="rgb",
    class_mode=None,
    batch_size=1,
    shuffle=False
)

xception = tf.keras.applications.Xception(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(input_size, input_size, 3),
    pooling='max',
)


modified_xception = Sequential()
modified_xception.add(xception)
modified_xception.add(Dense(units = 1, activation = 'sigmoid'))
modified_xception.summary()


# xception_json = open('/home/maria/PycharmProjects/DeepFake-Detect/tmp_checkpoint/xception_model.json', 'r')
# loaded_xception_json = xception_json.read()
# xception_json.close()
#
# modified_xception = model_from_json(loaded_xception_json)
# modified_xception.load_weights("/home/maria/PycharmProjects/DeepFake-Detect/tmp_checkpoint/xception_model.h5")
#
# modified_xception.summary()

# Compile model
modified_xception.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

checkpoint_filepath = './tmp_checkpoint'
os.makedirs(checkpoint_filepath, exist_ok=True)

custom_callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(checkpoint_filepath, 'xception_model.h5'),
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_best_only=True
    )
]

# Train network
num_epochs = 50
history = modified_xception.fit(
    train_generator,
    epochs=num_epochs,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=custom_callbacks
)
print(history.history)


# Plot results
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(1, len(acc) + 1)
#
# plt.plot(epochs, acc, 'bo', label='Training Accuracy')
# plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()
# plt.figure()
#
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
#
# plt.show()


# load the saved model that is considered the best
best_model_xception = load_model(os.path.join(checkpoint_filepath, 'xception_model.h5'))

xception_model_json = best_model_xception.to_json()
with open("./tmp_checkpoint/xception_model.json", "w") as json_file:
    json_file.write(xception_model_json)

# Generate predictions
test_generator.reset()

predicts = best_model_xception.predict(
    test_generator,
    verbose=1
)

test_results = pd.DataFrame({
    "Filename": test_generator.filenames,
    "Prediction": predicts.flatten()
})

print(test_results)
