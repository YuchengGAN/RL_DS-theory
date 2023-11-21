# coding=utf-8
import argparse, os
import tensorflow as tf
from PIL import Image
import keras.backend as K
import numpy as np
from keras.optimizers import Adam
# from keras.backend.tensorflow_backend import set_session
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from ICMDP_Env import ClassifyEnv
from get_model import get_text_model, get_image_model
from data_pre import load_data, get_imb_data
from tensorflow.keras.callbacks import ModelCheckpoint

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser()
parser.add_argument('--data', choices=['mnist', 'cifar10', 'famnist', 'imdb', 'pneumonia'], default='pneumonia')
parser.add_argument('--model', choices=['image', 'text'], default='image')
parser.add_argument('--imb-rate', type=float, default=0.35)  # (1038+1341)/(2431+2648)
# parser.add_argument('--min-class', type=str, default='456')
# parser.add_argument('--maj-class', type=str, default='789')
parser.add_argument('--min-class', type=str, default='23')  # peu_viral + Covid19
parser.add_argument('--maj-class', type=str, default='01')  # normal + peu_bacterial
parser.add_argument('--training-steps', type=int, default=120000)
args = parser.parse_args()
data_name = args.data

# 'Normal' : '0', 2648, 295
# 'Pneumonia-Bacterial': '1', 2431, 270
# 'Pneumonia-Viral' : '2', 1341, 149
# 'COVID-19' : '3', 1038, 115
x_train, y_train, x_test, y_test = load_data(data_name)
print(x_train.shape)  # (7458, 400, 300, 3)
print(y_train.shape)  # (7458,)
print(x_test.shape)  # (829, 400, 300, 3)
print(y_test.shape)  # (829,)
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
    model = get_image_model(in_shape, num_classes)
else:
    in_shape = [5000, 500]
    model = get_text_model(in_shape, num_classes)

batch_size = 16
INPUT_SHAPE = in_shape
# print(INPUT_SHAPE)  # (400, 300, 3)
# INPUT_SHAPE = 32
print(model.summary())


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
        # print('batch_size: ', batch.shape)  # (1, 400, 300, 3)
        processed_batch = batch.astype('float32') / 1.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


memory = SequentialMemory(limit=100000, window_length=1)
processor = ClassifyProcessor()
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=100000)
dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory, batch_size=batch_size,
               processor=processor, nb_steps_warmup=50000, gamma=0.5, target_model_update=10000,
               train_interval=4, delta_clip=1.)
dqn.compile(optimizer=Adam(lr=0.001), metrics=['mae'])


dqn_filepath = 'D:\Download_files\Evid_code\DRL-For-imbalanced-Classification-2\checkpoint\dqn_model.h5'
model_filepath = 'D:\Download_files\Evid_code\DRL-For-imbalanced-Classification-2\checkpoint\dqn_network_model.h5'
# please define our own filepath to save the weights of the probabilistic FitNet-4 classifier
os.makedirs(os.path.dirname(dqn_filepath), exist_ok=True)
os.makedirs(os.path.dirname(model_filepath), exist_ok=True)
# checkpoint_callback = ModelCheckpoint(
#     filepath=filepath, monitor='val_accuracy', verbose=1,
#     save_best_only=True, save_weights_only=True, save_frequency=1)

# dqn.fit(env, nb_steps=training_steps, log_interval=30000)
# dqn.save_weights(filepath=dqn_filepath)
# dqn.save_weights(filepath=model_filepath)
dqn.load_weights(filepath=dqn_filepath)
model.load_weights(filepath=model_filepath)

y_pred = model.predict(x_train)
print(y_pred.shape)
for _ in range(10):
    print(y_pred[_])
    print(y_train[_])
# [0.17706777 0.8229323 ]
# 1
# [1.0000000e+00 2.8789078e-23]
# 0

env.mode = 'test'
dqn.test(env, nb_episodes=1, visualize=False)
env = ClassifyEnv(mode, imb_rate, x_test, y_test)
env.mode = 'test'
dqn.test(env, nb_episodes=1, visualize=False)

