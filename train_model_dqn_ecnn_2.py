# coding=utf-8
import argparse, os
import tensorflow as tf
from PIL import Image
import keras.backend as K
import numpy as np
from keras.optimizers import Adam, Nadam
# from keras.backend.tensorflow_backend import set_session
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from ICMDP_Env_2 import ClassifyEnv
from get_model import get_text_model, get_image_mid_model, get_image_ecnn_model, get_image_parti_model
from data_pre import load_data, get_imb_data
from keras.utils import np_utils
from tensorflow.keras.callbacks import ModelCheckpoint

from libs import ds_layer  # Dempster-Shafer layer
from libs import utility_layer_train  # Utility layer for training
from libs import utility_layer_test  # Utility layer for training
from libs import AU_imprecision  # Metric average utility for set-valued classification

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.compat.v1.enable_eager_execution()


parser = argparse.ArgumentParser()
parser.add_argument('--data', choices=['mnist', 'cifar10', 'famnist', 'imdb', 'pneumonia'], default='pneumonia')
parser.add_argument('--model', choices=['image', 'text'], default='image')
parser.add_argument('--imb-rate', type=float, default=0.35)  # 0.26 / 0.74
# parser.add_argument('--min-class', type=str, default='456')
# parser.add_argument('--maj-class', type=str, default='789')
parser.add_argument('--min-class', type=str, default='23')  # peu_viral + Covid19
parser.add_argument('--maj-class', type=str, default='01')  # normal + peu_bacterial
parser.add_argument('--training-steps', type=int, default=120000)
args = parser.parse_args()
data_name = args.data
prototypes = 400


x_train, y_train, x_test, y_test = load_data(data_name)
y_train_cate = y_train.copy()
y_test_cate = y_test.copy()
y_train_cate = np_utils.to_categorical(y_train_cate)
y_test_cate = np_utils.to_categorical(y_test_cate)
print(x_train.shape)  # (7458, 64, 64, 3)
print(y_train.shape)  # (7458,)
print(x_test.shape)  # (829, 64, 64, 3)
print(y_test.shape)  # (829,)
print(y_train_cate.shape)
print(y_test_cate.shape)
imb_rate = args.imb_rate
maj_class = list(map(int, list(args.maj_class)))
min_class = list(map(int, list(args.min_class)))
# maj_class = int(args.maj_class)
# min_class = int(args.min_class)

# x_train, y_train, x_test, y_test = get_imb_data(x_train, y_train, x_test, y_test, imb_rate, min_class, maj_class)
print('x_train.shape: ', x_train.shape, 'y_train.shape: ', y_train.shape)
in_shape = x_train.shape[1:]
num_classes = len(set(y_test))
mode = 'train'
env = ClassifyEnv(mode, imb_rate, x_train, y_train)
nb_actions = num_classes
training_steps = args.training_steps
if args.model == 'image':
    model_e = get_image_ecnn_model(in_shape, num_classes)
else:
    in_shape = [5000, 500]
    model_e = get_text_model(in_shape, num_classes)

INPUT_SHAPE = in_shape
print(model_e.summary())


class ClassifyProcessor(Processor):
    def process_observation(self, observation):
        if args.model == 'text':
            return observation
        img = observation.reshape(INPUT_SHAPE)
        processed_observation = np.array(img)
        return processed_observation

    def process_state_batch(self, batch):
        if args.model == 'text':
            return batch.reshape((-1, INPUT_SHAPE[1]))
        batch = batch.reshape((-1,) + INPUT_SHAPE)
        processed_batch = batch.astype('float32') / 1.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


dqn_filepath = 'D:\Download_files\Evid_code\DRL-For-imbalanced-Classification-2\checkpoint\dqn_model.h5'
model_filepath = 'D:\Download_files\Evid_code\DRL-For-imbalanced-Classification-2\checkpoint\dqn_network_model.h5'
# dqn_filepath = 'D:\Download_files\Evid_code\DRL-For-imbalanced-Classification-master\checkpoint\dqn_model_2.h5'
# model_filepath = 'D:\Download_files\Evid_code\DRL-For-imbalanced-Classification-master\checkpoint\dqn_network_model_2.h5'
# please define our own filepath to save the weights of the probabilistic FitNet-4 classifier
os.makedirs(os.path.dirname(dqn_filepath), exist_ok=True)
os.makedirs(os.path.dirname(model_filepath), exist_ok=True)
# checkpoint_callback = ModelCheckpoint(
#     filepath=filepath, monitor='val_accuracy', verbose=1,
#     save_best_only=True, save_weights_only=True, save_frequency=1)

# model_e.load_weights(filepath=model_filepath, by_name=True)
# dqn.load_weights(filepath=dqn_filepath)
''''''
# # please give our own filepath to save the weights of the probabilistic FitNet-4 classifier
# feature = get_image_parti_model(in_shape)
# x_train_feature = feature.predict(x_train)
# # x_test_feature = feature.predict(x_test)
#
# # Use the features to train DS layer
# model_mid = get_image_mid_model(num_classes)
# # model_mid.compile(optimizer=Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),  # 0.001
# #                   loss='CategoricalCrossentropy',
# #                   metrics=['accuracy'])
# model_mid.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['mae'])
# model_mid.fit(x_train_feature, y_train_cate, batch_size=25,  epochs=2, verbose=1, shuffle=True)
# ''' y_train_cate -> y_train'''
# model_mid.summary()
''''''
# give the trained paramters to the evidential model
model_e.load_weights(filepath=model_filepath, by_name=True)
# ds_l1 = model_e.get_layer("d_s1")
# ds_l1_index = model_e.layers.index(ds_l1)
# print("ds_l1_index: ", ds_l1_index)
''''''
# ds_mid = model_mid.get_layer("d_s1_1")
# ds_mid_index = model_mid.layers.index(ds_mid)
# print("ds_mid_index: ", ds_mid_index)
# # please give our own filepath to save the weights of the probabilistic FitNet-4 classifier
# # DSLAYER_DS1_W = tf.reshape(model_mid.layers[ds_mid_index].get_weights()[0], [1, prototypes, 256])
# DSLAYER_DS1_W = model_mid.layers[ds_mid_index].get_weights()
# DSLAYER_DS1_activate_W = model_mid.layers[ds_mid_index+1].get_weights()
# DSLAYER_DS2_W = model_mid.layers[ds_mid_index+2].get_weights()
#
# model_e.layers[ds_l1_index].set_weights(DSLAYER_DS1_W)
# model_e.layers[ds_l1_index+1].set_weights(DSLAYER_DS1_activate_W)
# model_e.layers[ds_l1_index+2].set_weights(DSLAYER_DS2_W)
''''''
# final_feature = model_e.get_layer("dm").output
# new_layer = tf.keras.layers.Dense(name="Onehot2num", units=num_classes, activation='softmax')(final_feature)
# model_E = tf.keras.models.Model(name="E-CNN", inputs=model_e.input, outputs=new_layer)
# model_E.summary()
''''''


# fine-tune the golable weights in the evidential CNN classifier and evaluate the classifier
ecnn_dqn_filepath = 'D:\Download_files\Evid_code\DRL-For-imbalanced-Classification-2\checkpoint\ecnn_dqn_model__200_270000.h5'
ecnn_model_filepath = 'D:\Download_files\Evid_code\DRL-For-imbalanced-Classification-2\checkpoint\ecnn_dqn_network_model__200_270000.h5'
# define our own path to save the weights of the evidential FitNet-4 classifier
os.makedirs(os.path.dirname(ecnn_dqn_filepath), exist_ok=True)
os.makedirs(os.path.dirname(ecnn_model_filepath), exist_ok=True)
# checkpoint_callback = ModelCheckpoint(
#     ecnn_model_filepath, monitor='val_accuracy', verbose=1,
#     save_best_only=True, save_weights_only=True,
#     save_frequency=1)
# model_e.fit(x_train, y_train, batch_size=25,  epochs=3, verbose=1, callbacks=[checkpoint_callback],
#             validation_data=(x_test, y_test), shuffle=True)


memory = SequentialMemory(limit=100000, window_length=1)
processor = ClassifyProcessor()
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=100000)
dqn = DQNAgent(model=model_e, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=0.5, target_model_update=10000,
               train_interval=4, delta_clip=1.)


dqn.compile(optimizer=Adam(lr=0.001), metrics=['mae'])

# dqn.fit(env, nb_steps=training_steps, log_interval=30000)
# dqn.save_weights(filepath=ecnn_dqn_filepath)
# model_e.save_weights(filepath=ecnn_model_filepath)
model_e.load_weights(filepath=ecnn_model_filepath, by_name=True)
dqn.load_weights(filepath=ecnn_dqn_filepath)


y_pred = model_e.predict(x_train)
print(y_pred.shape)
for _ in range(10):
    print(y_pred[_])
    print(y_train[_])
    # [9.999959e-01 4.117716e-06]
    # 1
    print(type(y_pred[_][0]))

# y_pred = model_E.predict(x_train)
# print(y_pred.shape)
# for _ in range(10):
#     print(y_pred[_])
#     print(y_train[_])
# [0.10311303 0.89688694]
# 1
# [0.10415743 0.89584255]
# 0
# [0.10321393 0.8967861 ]
# 0
# [0.10318037 0.89681965]
# 0


env.mode = 'test'
dqn.test(env, nb_episodes=1, visualize=False)
env = ClassifyEnv(mode, imb_rate, x_test, y_test)
env.mode = 'test'
dqn.test(env, nb_episodes=1, visualize=False)

