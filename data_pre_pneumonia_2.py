import datetime
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score

from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pickle

# x_train_data.shape:  (5216, 64, 64, 3)
# y_train_data.shape:  (5216,)
# x_test_data.shape:  (624, 64, 64, 3)
# y_test_data.shape:  (624,)

# x_train_data.type:  <class 'numpy.ndarray'>
# y_train_data.type:  <class 'numpy.int32'>

BATCH_SIZE = 32
EPOCHS = 30
IM_SIZE_W = 64
IM_SIZE_H = 64

AUTOTUNE = tf.data.experimental.AUTOTUNE

tf.random.set_seed(10)

# device_name = tf.test.gpu_device_name()
# if "GPU" not in device_name:
#     print("GPU device not found")
# print('Found GPU at: {}'.format(device_name))


# for dirname, _, filenames in os.walk('../DRL-For-imbalanced-Classification-2/'):
#     print(dirname)

# All filenames
filenames = tf.io.gfile.glob('../DRL-For-imbalanced-Classification-2/Curated_X-Ray_Dataset_4/*/*')
print(len(filenames))
print(filenames[:3])

# To DataFrame
data = pd.DataFrame()
for el in range(0, len(filenames)):
    target = filenames[el].split('\\')[-2]
    path = filenames[el]

    data.loc[el, 'filename'] = path
    data.loc[el, 'class'] = target

print(data['class'].value_counts(dropna=False))
print(data)


# Shuffle Data
data = shuffle(data, random_state=42)
data.reset_index(drop=True, inplace=True)
print(data)

change = {
    'Normal': 0,
    'Pneumonia-Bacterial': 1,
    'Pneumonia-Viral': 2,
    'COVID-19': 3,
}
data['class'] = data['class'].map(change)
print(data)
# print(type(data['class'][2]))  # <class 'numpy.int64'>


# Drop out trash
indexes = []


def func(x):
    if x[-4:] != '.jpg':
        idx = data[data['filename'] == x].index
        indexes.append(idx[0])
        print(idx[0], x)
    return x


data['filename'].map(func)
print(data.shape)
data.drop(index=indexes, axis=0, inplace=True)
data.reset_index(drop=True, inplace=True)
print(data.shape)  # (9208, 2)
data_shape = data.shape


# SPLIT train_data, val_data
train_data, val_data = train_test_split(data, test_size=0.1, random_state=42, stratify=data['class'])
print(train_data['class'].value_counts(dropna=False))
print(val_data['class'].value_counts(dropna=False))

# SPLIT train_data, test_data
train_data, test_data = train_test_split(train_data, test_size=0.1, random_state=42, stratify=train_data['class'])
print(train_data['class'].value_counts(dropna=False))
print(test_data['class'].value_counts(dropna=False))
print('train_data.shape: ', train_data.shape)  # (7458, 2)
print('test_data.shape: ', test_data.shape)  # (829, 2)

# print(type(train_data))  # <class 'pandas.core.frame.DataFrame'>
# print(train_data)
train_data = train_data.reset_index()
test_data = test_data.reset_index()

y_train = train_data['class'].to_numpy()
y_test = test_data['class'].to_numpy()
print('y_train.shape: ', y_train.shape)
print('y_test.shape: ', y_test.shape)
# print(y_train)  # [0 2 3 ... 0 3 2]
# print(y_train.shape)  # (7458,)
# print(type(y_train))  # <class 'numpy.ndarray'>
# print(type(y_train[0]))  # <class 'numpy.int64'>

# data_output = open('./pneumonia_y_train_4_64_64.pkl', 'wb')
# pickle.dump(y_train, data_output)
# data_output.close()
#
# data_output = open('./pneumonia_y_test_4_64_64.pkl', 'wb')
# pickle.dump(y_test, data_output)
# data_output.close()


# train_images = []
# # Images shape
# for el in range(0, train_data.shape[0]):
#     path = train_data.loc[el, 'filename']
#     # print(type(path))  # <class 'str'>
#     path = path.replace("\\", "/")
#     # print(path)
#     img = tf.io.read_file(path)
#     img = tf.image.decode_jpeg(img, channels=3)
#     img = tf.image.resize(img, [IM_SIZE_H, IM_SIZE_W])
#     img = img / 255.0
#     img = img.numpy()
#     # print(img.shape)  # (400, 300, 3)
#     train_images.append(img[np.newaxis, :])
# train_images = np.concatenate(train_images, axis=0)
# x_train = train_images
# print('x_train.shape: ', x_train.shape)
#
# # wb 以二进制写入
# data_output = open('./pneumonia_x_train_4_64_64.pkl', 'wb')
# pickle.dump(x_train, data_output)
# data_output.close()
# train_images = None


# test_images = []
# # Images shape
# for el in range(0, test_data.shape[0]):
#     path = test_data.loc[el, 'filename']
#     # print(type(path))  # <class 'str'>
#     path = path.replace("\\", "/")
#     # print(path)
#     img = tf.io.read_file(path)
#     img = tf.image.decode_jpeg(img, channels=3)
#     img = tf.image.resize(img, [IM_SIZE_H, IM_SIZE_W])
#     img = img / 255.0
#     img = img.numpy()
#     # print(img.shape)  # (400, 300, 3)
#     test_images.append(img[np.newaxis, :])
# test_images = np.concatenate(test_images, axis=0)
# x_test = test_images
# print('x_test.shape: ', x_test.shape)
#
# data_output = open('./pneumonia_x_test_4_64_64.pkl', 'wb')
# pickle.dump(x_test, data_output)
# data_output.close()


# rb 以二进制读取
data_input = open('./pneumonia_x_train_4_64_64.pkl', 'rb')
read_data = pickle.load(data_input)
data_input.close()
print(read_data.shape)



