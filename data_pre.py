# coding=utf-8
# from keras.utils import to_categorical
from keras.datasets import mnist, fashion_mnist, cifar10, imdb
import random
from sklearn.metrics import confusion_matrix
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import os
# from data_pre import load_data
import tensorflow as tf
import pathlib
import random
import pickle


def load_famnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    y_train = y_train.reshape(y_train.shape[0], )
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_test = y_test.reshape(y_test.shape[0], )
    x_train = x_train / 255.
    x_test = x_test / 255.
    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    y_train = y_train.reshape(y_train.shape[0], )
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_test = y_test.reshape(y_test.shape[0], )
    x_train = x_train / 255.
    x_test = x_test / 255.
    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test


def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(-1, 32, 32, 3)
    y_train = y_train.reshape(y_train.shape[0], )
    x_test = x_test.reshape(-1, 32, 32, 3)
    y_test = y_test.reshape(y_test.shape[0], )
    x_train = x_train / 255.
    x_test = x_test / 255.
    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test


def load_imdb():
    config = [5000, 500]
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=config[0])
    x_train = pad_sequences(x_train, maxlen=config[1])
    x_test = pad_sequences(x_test, maxlen=config[1])
    return x_train, y_train, x_test, y_test


def load_pneu():
    data_input = open('./pneumonia_x_train_4_64_64.pkl', 'rb')
    x_train = pickle.load(data_input)
    data_input.close()

    data_input = open('./pneumonia_x_test_4_64_64.pkl', 'rb')
    x_test = pickle.load(data_input)
    data_input.close()

    data_input = open('./pneumonia_y_train_4_64_64.pkl', 'rb')
    y_train = pickle.load(data_input)
    data_input.close()

    data_input = open('./pneumonia_y_test_4_64_64.pkl', 'rb')
    y_test = pickle.load(data_input)
    data_input.close()

    return x_train, y_train, x_test, y_test


def load_data(data_name):
    if data_name == 'famnist':
        x_train, y_train, x_test, y_test = load_famnist()
    elif data_name == 'mnist':
        x_train, y_train, x_test, y_test = load_mnist()
    elif data_name == 'cifar10':
        x_train, y_train, x_test, y_test = load_cifar10()
    elif data_name == 'pneumonia':
        x_train, y_train, x_test, y_test = load_pneu()
    else:
        x_train, y_train, x_test, y_test = load_imdb()

    return x_train, y_train, x_test, y_test


def get_imb_data(x_train, y_train, x_test, y_test, imb_rate, min_class, maj_class):
    maj_x_train = []
    maj_y_train = []
    min_x_train = []
    min_y_train = []
    #     print(min_class, maj_class)
    for i in range(len(y_train)):
        # if int(y_train[i]) == min_class:
        if y_train[i] in min_class:
            min_x_train.append(x_train[i])
            min_y_train.append(y_train[i])

        # if int(y_train[i]) == maj_class:
        if y_train[i] in maj_class:
            maj_x_train.append(x_train[i])
            maj_y_train.append(y_train[i])
    #
    min_len = int(len(maj_y_train) * imb_rate)

    random.seed(66)
    random.shuffle(min_x_train)
    random.seed(66)
    random.shuffle(min_y_train)

    new_x_train = maj_x_train + min_x_train[:min_len]
    new_y_train = maj_y_train + min_y_train[:min_len]
    #     print(len(new_y_train),len(new_y_train))
    #     print(len(maj_x_train))
    new_x_test = []
    new_y_test = []

    for i in range(len(y_test)):
        # if int(y_test[i]) == min_class:
        if y_test[i] in min_class:
            new_x_test.append(x_test[i])
            new_y_test.append(y_test[i])
        # if int(y_test[i]) == maj_class:
        if y_test[i] in maj_class:
            new_x_test.append(x_test[i])
            new_y_test.append(y_test[i])

    new_x_train = np.array(new_x_train)
    new_y_train = np.array(new_y_train)
    new_x_test = np.array(new_x_test)
    new_y_test = np.array(new_y_test)

    idx = [i for i in range(len(new_y_train))]
    np.random.seed(23)
    np.random.shuffle(idx)

    new_x_train = new_x_train[idx]
    new_y_train = new_y_train[idx]

    return new_x_train, new_y_train, new_x_test, new_y_test
