import numpy as np
import csv
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder


def read_data(path, shuffle=False):
    """
    read data set
    :param path: csv file path
    :param shuffle: whether to shuffle
    :return: the numpy array of data and labels
    """
    data = []
    with open(path) as f:
        lines = csv.reader(f)
        for line in lines:
            data.append(map(int, line))
    data = np.array(data)
    if shuffle:
        np.random.shuffle(data)
    label = data[:, 0]
    data = data[:, 1:]
    label = (label + 1) / 2
    return data, label


def data_aug_tf(data, label):
    """
    data augment tensor form
    :param data:
    :param label:
    :return: augmented data and labels
    """
    n, c = data.shape.as_list()
    label = tf.concat([label, 1 - label], axis=0)
    reverse_mask = tf.concat([tf.ones([n, 3]), -1 * tf.ones([n, 113])], axis=1)
    data_reverse = data * reverse_mask
    data = tf.concat([data, data_reverse], axis=0)
    return data, label


def data_aug_np(data, label):
    """
    data augment numpy form
    :param data:
    :param label:
    :return: augmented data and labels
    """
    n, c = data.shape
    label = np.concatenate([label, 1 - label], axis=0)
    reverse_mask = np.concatenate([np.ones([n, 3]), -1 * np.ones([n, 113])], axis=1)
    data_reverse = data * reverse_mask
    data = np.concatenate([data, data_reverse], axis=0)
    return data, label


def onehot_encode(*data_list):
    """
    one-hot encode the features
    :param data_list: all arrays that need to be one-hot encoded
    :return: arrays that is one-hot encoded
    """
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
        ret_data.append(data[left_idx: right_idx, :].astype(np.float32))
        left_idx = right_idx
    return ret_data
