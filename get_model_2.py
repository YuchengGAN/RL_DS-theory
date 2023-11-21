# coding=utf-8
import keras
import tensorflow as tf
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Activation, Embedding, Lambda
from keras.optimizers import Adam, SGD
from keras.layers import LSTM

from libs import ds_layer  # Dempster-Shafer layer
from libs import utility_layer_train  # Utility layer for training
from libs import utility_layer_test  # Utility layer for training
from libs import AU_imprecision  # Metric average utility for set-valued classification

tf.keras.utils.get_custom_objects()['DS1'] = ds_layer.DS1
tf.keras.utils.get_custom_objects()['DS1_activate'] = ds_layer.DS1_activate
tf.keras.utils.get_custom_objects()['DS2'] = ds_layer.DS2
tf.keras.utils.get_custom_objects()['DS2_omega'] = ds_layer.DS2_omega
tf.keras.utils.get_custom_objects()['DS3_Dempster'] = ds_layer.DS3_Dempster
tf.keras.utils.get_custom_objects()['DS3_normalize'] = ds_layer.DS3_normalize
tf.keras.utils.get_custom_objects()['DM'] = utility_layer_train.DM


def get_text_model(input_shape, output):
    top_words, max_words = input_shape
    model = Sequential()
    model.add(Embedding(top_words, 128, input_length=max_words))
    model.add(Flatten())
    model.add(Dense(250))
    model.add(Activation('relu'))
    model.add(Dense(output))
    return model


# def get_image_model(in_shape, output):
#     model = Sequential()
#     model.add(Conv2D(32, (5, 5), padding='Same', input_shape=in_shape))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Conv2D(32, (5, 5), padding='Same'))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Flatten())
#     model.add(Dense(256))
#     model.add(Activation('relu'))
#     model.add(Dense(output))
#     return model


def get_image_model(in_shape, output):
    inputs = tf.keras.layers.Input(in_shape)
    c1_1 = tf.keras.layers.Conv2D(128, (5, 5), activation='relu', padding='Same')(inputs)
    p1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c1_1)
    d1 = tf.keras.layers.Dropout(0.1)(p1)
    c1_2 = tf.keras.layers.Conv2D(128, (5, 5), activation='relu', padding='Same')(d1)
    p2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c1_2)
    d2 = tf.keras.layers.Dropout(0.1)(p2)
    c1_3 = tf.keras.layers.Conv2D(128, (5, 5), activation='relu', padding='Same')(d2)
    p3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c1_3)
    d3 = tf.keras.layers.Dropout(0.1)(p3)
    c1_4 = tf.keras.layers.Conv2D(256, (5, 5), activation='relu', padding='Same')(d3)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c1_4)
    d4 = tf.keras.layers.Dropout(0.1)(p4)
    flatten1 = tf.keras.layers.Flatten()(d4)
    den1 = tf.keras.layers.Dense(1024, activation='relu')(flatten1)
    d5 = tf.keras.layers.Dropout(0.1)(den1)
    outputs = tf.keras.layers.Dense(output, activation='softmax')(d5)
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model


def get_image_parti_model(in_shape):
    inputs = tf.keras.layers.Input(in_shape)
    c1_1 = tf.keras.layers.Conv2D(128, (5, 5), activation='relu', padding='Same')(inputs)
    p1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c1_1)
    d1 = tf.keras.layers.Dropout(0.1)(p1)
    c1_2 = tf.keras.layers.Conv2D(128, (5, 5), activation='relu', padding='Same')(d1)
    p2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c1_2)
    d2 = tf.keras.layers.Dropout(0.1)(p2)
    c1_3 = tf.keras.layers.Conv2D(128, (5, 5), activation='relu', padding='Same')(d2)
    p3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c1_3)
    d3 = tf.keras.layers.Dropout(0.1)(p3)
    c1_4 = tf.keras.layers.Conv2D(256, (5, 5), activation='relu', padding='Same')(d3)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c1_4)
    d4 = tf.keras.layers.Dropout(0.1)(p4)
    flatten1 = tf.keras.layers.Flatten()(d4)
    # d1 = tf.keras.layers.Dense(256, activation='relu')(flatten1)
    den1 = tf.keras.layers.Dense(1024, activation='relu')(flatten1)
    d5 = tf.keras.layers.Dropout(0.1)(den1)
    model = tf.keras.Model(inputs=[inputs], outputs=[d5])
    return model


# def get_image_ecnn_model(in_shape, output):
#     prototypes = 200
#     model = Sequential(name="E-CNN")
#     model.add(Conv2D(32, (5, 5), padding='Same', input_shape=in_shape))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Conv2D(32, (5, 5), padding='Same'))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Flatten())
#     model.add(Dense(256))
#     model.add(Activation('relu'))
#     # model.add(Dense(output))
#     model.add(ds_layer.DS1(prototypes, 256))
#     model.add(ds_layer.DS1_activate(prototypes))
#     model.add(ds_layer.DS2(prototypes, output))
#     model.add(ds_layer.DS2_omega(prototypes, output))
#     model.add(ds_layer.DS3_Dempster(prototypes, output))
#     model.add(ds_layer.DS3_normalize())
#     model.add(utility_layer_train.DM(0.9, output))
#     return model


def get_image_ecnn_model(in_shape, output):
    prototypes = 300
    inputs = tf.keras.layers.Input(in_shape)
    c1_1 = tf.keras.layers.Conv2D(128, (5, 5), activation='relu', padding='Same')(inputs)
    p1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c1_1)
    d1 = tf.keras.layers.Dropout(0.1)(p1)
    c1_2 = tf.keras.layers.Conv2D(128, (5, 5), activation='relu', padding='Same')(d1)
    p2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c1_2)
    d2 = tf.keras.layers.Dropout(0.1)(p2)
    c1_3 = tf.keras.layers.Conv2D(128, (5, 5), activation='relu', padding='Same')(d2)
    p3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c1_3)
    d3 = tf.keras.layers.Dropout(0.1)(p3)
    c1_4 = tf.keras.layers.Conv2D(256, (5, 5), activation='relu', padding='Same')(d3)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c1_4)
    d4 = tf.keras.layers.Dropout(0.1)(p4)
    flatten1 = tf.keras.layers.Flatten()(d4)
    # den1 = tf.keras.layers.Dense(256, activation='relu')(flatten1)
    # outputs = tf.keras.layers.Dense(output, activation='softmax')(den1)
    den1 = tf.keras.layers.Dense(1024, activation='relu')(flatten1)
    d5 = tf.keras.layers.Dropout(0.1)(den1)
    # DS layer
    ED = ds_layer.DS1(prototypes, 1024)(d5)  # 1288对应d5层输出层元素个数
    ED_ac = ds_layer.DS1_activate(prototypes)(ED)
    mass_prototypes = ds_layer.DS2(prototypes, output)(ED_ac)
    mass_prototypes_omega = ds_layer.DS2_omega(prototypes, output)(mass_prototypes)
    mass_Dempster = ds_layer.DS3_Dempster(prototypes, output)(mass_prototypes_omega)
    mass_Dempster_normalize = ds_layer.DS3_normalize()(mass_Dempster)
    # Utility layer for training
    outputs = utility_layer_train.DM(0.9, output)(mass_Dempster_normalize)
    model = tf.keras.Model(name="e-CNN", inputs=[inputs], outputs=[outputs])
    return model


def get_image_mid_model(output):
    prototypes = 200
    inputs = tf.keras.layers.Input(1024)
    ED = ds_layer.DS1(prototypes, 1024)(inputs)
    ED_ac = ds_layer.DS1_activate(prototypes)(ED)
    mass_prototypes = ds_layer.DS2(prototypes, output)(ED_ac)
    mass_prototypes_omega = ds_layer.DS2_omega(prototypes, output)(mass_prototypes)
    mass_Dempster = ds_layer.DS3_Dempster(prototypes, output)(mass_prototypes_omega)
    mass_Dempster_normalize = ds_layer.DS3_normalize()(mass_Dempster)
    # Utility layer for training
    outputs = utility_layer_train.DM(0.9, output)(mass_Dempster_normalize)
    model = tf.keras.Model(name="mid-E-CNN", inputs=[inputs], outputs=[outputs])
    return model

