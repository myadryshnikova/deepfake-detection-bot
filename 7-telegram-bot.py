import glob
import math
import os
import random

import cv2
import numpy as np
import telebot
from tensorflow import keras
import tensorflow.keras.optimizers as Optimizer
from tensorflow.keras.models import model_from_json
from mtcnn import MTCNN
from telebot import types


bot = telebot.TeleBot(TOKEN)


def load_resnet_model():
    resnet_json = open('/home/maria/PycharmProjects/DeepFake-Detect/tmp_checkpoint/resnet_model.json', 'r')
    loaded_resnet_json = resnet_json.read()
    resnet_json.close()

    loaded_resnet_model = model_from_json(loaded_resnet_json)
    loaded_resnet_model.load_weights("/home/maria/PycharmProjects/DeepFake-Detect/tmp_checkpoint/resnet_model.h5")

    loaded_resnet_model.compile(optimizer=Optimizer.Adam(learning_rate=0.0001),
                                loss='binary_crossentropy',
                                metrics=['accuracy'])

    return loaded_resnet_model


def load_xception_model():
    xception_json = open('/home/maria/PycharmProjects/DeepFake-Detect/tmp_checkpoint/xception_model.json', 'r')
    loaded_xception_json = xception_json.read()
    xception_json.close()

    loaded_xception_model = model_from_json(loaded_xception_json)
    loaded_xception_model.load_weights("/home/maria/PycharmProjects/DeepFake-Detect/tmp_checkpoint/xception_model.h5")

    loaded_xception_model.compile(optimizer=Optimizer.Adam(learning_rate=0.0001),
                                  loss='binary_crossentropy',
                                  metrics=['accuracy'])

    return loaded_xception_model


def get_label(predicts_resnet, predicts_xception):
    predict_resnet = np.median(predicts_resnet)
    predict_xception = np.median(predicts_xception)

    predict = (predict_resnet + predict_xception) / 2

    print(f'PREDICT RESNET: {predict_resnet}')
    print(f'PREDICT XCEPTION: {predict_xception}')
    print(f'SUMMARY PREDICT: {predict}')

    if predict_resnet > 0.5:
        label = 'реальное видео'
        confidence = round(predict_resnet * 100, 2)
    else:
        label = 'фейк'
        confidence = round((1 - predict_resnet) * 100, 2)

    return label, confidence


def random_video(code, message):
    path = "/home/maria/PycharmProjects/DeepFake-Detect/random_videos"

    video = random.choice(os.listdir(os.path.join(path, code)))

    bot.send_video(message.chat.id, open(os.path.join(path, code, video), 'rb'))


def save_video_from_bot_to_server(message):
    fileID = message.video.file_id
    file_info = bot.get_file(fileID)
    downloaded_file = bot.download_file(file_info.file_path)

    with open("/home/maria/PycharmProjects/DeepFake-Detect/video/video.mp4", 'wb') as new_file:
        new_file.write(downloaded_file)


def delete_files_from_server():
    for file in glob.glob("/home/maria/PycharmProjects/DeepFake-Detect/video/*"):
        if file == '/home/maria/PycharmProjects/DeepFake-Detect/video/faces':
            continue
        os.remove(file)

    for face in glob.glob("/home/maria/PycharmProjects/DeepFake-Detect/video/faces/*"):
        os.remove(face)


def convert_video_to_images():
    count = 0
    filename = "video.mp4"
    tmp_path = os.path.join('/home/maria/PycharmProjects/DeepFake-Detect/', filename[:5])
    os.makedirs(tmp_path, exist_ok=True)
    os.chdir(tmp_path)

    cap = cv2.VideoCapture(filename)

    frame_rate = cap.get(5)

    while cap.isOpened():
        frame_id = cap.get(1)
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

            new_filename = '{}-{:03d}.png'.format(filename, count)
            count = count + 1
            cv2.imwrite(new_filename, new_frame)
    cap.release()


def crop_faces_from_images():
    path = '/home/maria/PycharmProjects/DeepFake-Detect/video'
    faces_path = path + '/faces'
    #os.makedirs(faces_path, exist_ok=True)
    listOfFiles = os.listdir(path)
    print(listOfFiles)
    listOfFiles.remove('faces')
    listOfFiles.remove('video.mp4')
    print(listOfFiles)

    for filename in listOfFiles:
        os.chdir(faces_path)
        detector = MTCNN()
        image = cv2.cvtColor(cv2.imread(os.path.join(path, filename)), cv2.COLOR_BGR2RGB)
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
                new_filename = '{}-{:02d}.png'.format(os.path.join(faces_path, filename), count)
                count = count + 1
                cv2.imwrite(new_filename, cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR))


def download_video_and_preprocess():
    convert_video_to_images()
    crop_faces_from_images()

    images = []

    path = '/home/maria/PycharmProjects/DeepFake-Detect/video/faces/'
    listOfFiles = os.listdir(path[:-1])

    for filename in listOfFiles:
        image = cv2.imread(path + filename)
        image = cv2.resize(image, (128, 128))
        image = np.array(image)
        image = image / 255
        images.append(image)

    images = np.array(images)

    return images


def detection(message):
    images = download_video_and_preprocess()
    model_resnet = load_resnet_model()
    model_xception = load_xception_model()

    predicts_resnet = []
    predicts_xception = []

    for image in images:
        image = image[None, ...]
        predict_resnet = model_resnet.predict(image)
        predicts_resnet.append(predict_resnet)

        predict_xception = model_xception.predict(image)
        predicts_xception.append(predict_xception)

    print(f'ResNet predicts: {predicts_resnet}')
    print(f'Xception predicts: {predicts_xception}')

    pred_class, confidence = get_label(predicts_resnet, predicts_xception)

    delete_files_from_server()

    bot.send_message(message.chat.id, f'Думаю, это {pred_class}!\nЯ уверен в этом на {confidence} %.')


def retrain(message):
    download_video_and_preprocess()

    keyboard = types.InlineKeyboardMarkup()
    keyboard.row(telebot.types.InlineKeyboardButton('Фейк',
                                                    callback_data='class-fake'))
    keyboard.row(telebot.types.InlineKeyboardButton('Реальное видео',
                                                    callback_data='class-real'))

    send = bot.send_message(message.chat.id,
                            "А теперь нажми на кнопку класса картинки.",
                            reply_markup=keyboard)


def retraining_model(label, message):
    images = download_video_and_preprocess()

    if label == 'fake':
        label = 0
    else:
        label = 1

    labels = []
    for image in images:
        labels.append(label)

    labels = np.array(labels)

    resnet_model = load_resnet_model()
    resnet_model.fit(images, labels)
    resnet_model.save_weights("/home/maria/PycharmProjects/DeepFake-Detect/tmp_checkpoint/resnet_model.h5")

    xception_model = load_xception_model()
    xception_model.fit(images, labels)
    xception_model.save_weights("/home/maria/PycharmProjects/DeepFake-Detect/tmp_checkpoint/xception_model.h5")

    delete_files_from_server()

    bot.send_message(message.chat.id, text="Спасибо! Теперь я знаю чуть больше :)")


@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id,
                     'Привет!\n\nОтправь мне видео, и я подскажу тебе, дипфейк это или нет.\nЕщё ты можешь помочь мне дообучиться.\n\n<b>Пришли видео :)</b>\n Если у тебя нет видео, но ты хочешь меня потестировать, введи команду /get_video. Я подберу тебе видео из моей коллекции.',
                     parse_mode='HTML')


@bot.message_handler(commands=['get_video'])
def get_video(message):
    keyboard = types.InlineKeyboardMarkup()
    keyboard.row(telebot.types.InlineKeyboardButton('Дипфейка',
                                                    callback_data='get-deepfake'))
    keyboard.row(telebot.types.InlineKeyboardButton('Реальное видео',
                                                    callback_data='get-real-video'))

    bot.send_message(message.chat.id,
                     'Какое видео ты хочешь получить?',
                     reply_markup=keyboard)


@bot.message_handler(content_types=['video'])
def get_video(message):
    os.chdir("/home/maria")
    save_video_from_bot_to_server(message)

    keyboard = types.InlineKeyboardMarkup()
    keyboard.row(telebot.types.InlineKeyboardButton('Распознаем', callback_data='do-detection'))
    keyboard.row(telebot.types.InlineKeyboardButton('Дообучимся', callback_data='do-retrain'))

    bot.send_message(message.chat.id,
                     'Что сделаем?',
                     reply_markup=keyboard)


@bot.callback_query_handler(func=lambda call: True)
def callback(query):
    data = query.data

    if data.startswith('do-'):
        code = query.data[3:]

    elif data.startswith('class-'):
        code = query.data[6:]

    elif data.startswith('get-'):
        code = query.data[4:]

    get_callback(code, query)


def get_callback(code, query):
    bot.answer_callback_query(query.id)

    if code == 'detection':
        detection(query.message)
    elif code == 'retrain':
        retrain(query.message)
    elif code == 'deepfake' or code == 'real-video':
        random_video(code, query.message)
    else:
        retraining_model(code, query.message)


# def started_work_bot():
print('Started work bot.')
bot.infinity_polling()
