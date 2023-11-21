from data_pre import load_data, get_imb_data
import pickle
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data', choices=['mnist', 'cifar10', 'famnist', 'imdb', 'pneumonia'], default='pneumonia')
parser.add_argument('--model', choices=['image', 'text'], default='image')
parser.add_argument('--imb-rate', type=float, default=0.15)  #
# parser.add_argument('--min-class', type=str, default='456')
# parser.add_argument('--maj-class', type=str, default='789')
parser.add_argument('--min-class', type=str, default='23')  # peu_viral + Covid19
parser.add_argument('--maj-class', type=str, default='01')  # normal + peu_bacterial
parser.add_argument('--training-steps', type=int, default=120000)

args = parser.parse_args()
data_name = args.data
x_train, y_train, x_test, y_test = load_data(data_name)

print(x_train.shape)  # (5216, 64, 64, 3)
print(y_train.shape)  # (5216,)
print(x_test.shape)  # (624, 64, 64, 3)
print(y_test.shape)  # (624,)
imb_rate = args.imb_rate
maj_class = list(map(int, list(args.maj_class)))
min_class = list(map(int, list(args.min_class)))

# print(np.count_nonzero(y_train == 0))  # 2648
# print(np.count_nonzero(y_train == 1))  # 2431
# print(np.count_nonzero(y_train == 2))  # 1341
# print(np.count_nonzero(y_train == 3))  # 1038

# print(np.count_nonzero(y_test == 0))  # 295
# print(np.count_nonzero(y_test == 1))  # 270
# print(np.count_nonzero(y_test == 2))  # 149
# print(np.count_nonzero(y_test == 3))  # 115

x_train, y_train, x_test, y_test = get_imb_data(x_train, y_train, x_test, y_test, imb_rate, min_class, maj_class)
print('x_train.shape: ', x_train.shape, 'y_train.shape: ', y_train.shape)

print(np.count_nonzero(y_train == 0))  # 2648
print(np.count_nonzero(y_train == 1))  # 2431
print(np.count_nonzero(y_train == 2))  # 428
print(np.count_nonzero(y_train == 3))  # 333

print(np.count_nonzero(y_test == 0))  # 295
print(np.count_nonzero(y_test == 1))  # 270
print(np.count_nonzero(y_test == 2))  # 149
print(np.count_nonzero(y_test == 3))  # 115

print(y_train[0:100])
print(x_train[0])











