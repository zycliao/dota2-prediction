import numpy as np
import csv
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder


def read_data(path):
    data = []
    with open(path) as f:
        lines = csv.reader(f)
        for line in lines:
            data.append(map(int, line))
    data = np.array(data)
    label = data[:, 0]
    data = data[:, 1:]
    label = (label + 1) / 2
    return data, label


def data_aug(data, label):
    n, c = data.shape.as_list()
    label = tf.concat([label, 1 - label], axis=0)
    reverse_mask = tf.concat([tf.ones([n, 3]), -1 * tf.ones([n, 113])], axis=1)
    data_reverse = data * reverse_mask
    data = tf.concat([data, data_reverse], axis=0)
    return data, label


def onehot_encode(data_list):
    data_len = map(lambda x: x.shape[0], data_list)
    data = np.concatenate(data_list, axis=0)
    encoder1 = OneHotEncoder()
    encoder2 = OneHotEncoder()
    encoder3 = OneHotEncoder()
    a = encoder1.fit_transform(data[:, 0: 1]).toarray()
    b = encoder2.fit_transform(data[:, 1: 2]).toarray()
    c = encoder3.fit_transform(data[:, 2: 3]).toarray()
    data = np.concatenate([a, b, c, data[:, 3:]], axis=1)
    left_idx = 0
    ret_data = []
    for one_len in data_len:
        right_idx = left_idx + one_len
        ret_data.append(data[left_idx: right_idx, :])
        left_idx = right_idx
    return ret_data
