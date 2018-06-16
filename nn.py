import pickle
import time

import tensorflow as tf
from preprocessing import read_data, data_aug_tf, onehot_encode
import os
from evaluate import evaluate
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import l2_regularizer


def nn(data, label, wd_rate=None, training=True, reuse=False):
    """
    Construct the neural network
    :param data: input data
    :param label: 0/1 label
    :param wd_rate: weight decay rate
    :param training: whether training or not
    :param reuse: reuse this network
    :return:
    """
    with tf.variable_scope('fcnn', reuse=reuse,
                           initializer=xavier_initializer(),
                           regularizer=l2_regularizer(wd_rate) if training else None):
        hidden_nums = 100
        activation = tf.nn.sigmoid
        net = tf.layers.dense(data, hidden_nums, activation=activation)
        net = tf.layers.dense(net, hidden_nums, activation=activation)
        net = tf.layers.dense(net, hidden_nums, activation=activation)
        out = tf.layers.dense(net, 2)
        score = tf.nn.softmax(out)
        predict = tf.arg_max(out, 1, output_type=tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(label, predict), dtype=tf.float32))
        onehot_label = tf.one_hot(label, 2)
        if training:
            xent_loss = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=onehot_label)
            xent_loss = tf.reduce_mean(xent_loss)
            return accuracy, score, xent_loss
        else:
            return accuracy, score


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    lr = 1e-2
    wd = 1e-5
    val_num = 10000
    aug = True
    result_name = 'nn_noonehotencoder3feat_3fc_100'
    mode = 'all_feat'

    train_file = 'dota2Train.csv'
    test_file = 'dota2Test.csv'
    train_data, train_label = read_data(train_file, shuffle=True)
    test_data, test_label = read_data(test_file, shuffle=True)

    with tf.Graph().as_default():
        # build graph

        # control input
        batch_size = tf.placeholder(tf.int32, shape=[])

        # data input
        train_data = tf.constant(train_data, dtype=tf.float32, shape=train_data.shape)
        train_label = tf.constant(train_label, dtype=tf.int32, shape=train_label.shape)
        test_data = tf.constant(test_data, dtype=tf.float32, shape=test_data.shape)
        test_label = tf.constant(test_label, dtype=tf.int32, shape=test_label.shape)

        # split validation set
        if val_num:
            val_data = train_data[:val_num, :]
            val_label = train_label[:val_num]
            train_data = train_data[val_num:, :]
            train_label = train_label[val_num:]
        else:
            val_data, val_label = None, None

        # data augment
        if aug:
            train_data, train_label = data_aug_tf(train_data, train_label)
            test_data, test_label = data_aug_tf(test_data, test_label)
            if val_num:
                val_data, val_label = data_aug_tf(val_data, val_label)

        # feature process
        if mode == 'only_heroes':
            train_data = train_data[:, 3:]
            test_data = test_data[:, 3:]
            if val_num:
                val_data = val_data[:, 3:]
        elif mode == 'all_feat':
            pass
        elif mode == 'one_hot_all_feat':
            train_data, test_data, val_data = tf.py_func(onehot_encode, [train_data, test_data, val_data],
                                                         [tf.float32, tf.float32, tf.float32])
            train_data.set_shape(train_label.shape.as_list() + [172])
            test_data.set_shape(test_label.shape.as_list() + [172])
            val_data.set_shape(val_label.shape.as_list() + [172])


        # shuffle
        def select_batch(train_data, train_label):
            train_label = tf.expand_dims(tf.cast(train_label, dtype=tf.float32), axis=1)
            all_train = tf.concat([train_label, train_data], axis=1)
            all_train = tf.random_shuffle(all_train)
            train_data = all_train[:, 1:][:batch_size, :]
            train_label = tf.cast(all_train[:, 0], dtype=tf.int32)[:batch_size]
            return train_data, train_label


        train_data, train_label = tf.cond(batch_size > 0, lambda: select_batch(train_data, train_label),
                                          lambda: (train_data, train_label))

        # training graph
        train_acc, train_score, xent_loss = nn(train_data, train_label, wd_rate=wd)
        reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(xent_loss + reg_loss)

        # validation and test graph
        if val_num:
            val_acc, val_score = nn(val_data, val_label, training=False, reuse=True)
        test_acc, test_score = nn(test_data, test_label, training=False, reuse=True)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        start = time.time()
        with tf.Session(config=config) as sess:
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)
            print_freq = 30
            train_acc_ = 0
            xent_loss_ = 0
            reg_loss_ = 0
            for step in range(10000):
                # train
                ret = sess.run([train_acc, xent_loss, reg_loss, train_op],
                               feed_dict={batch_size: 100})
                train_acc_ += ret[0]
                xent_loss_ += ret[1]
                reg_loss_ += ret[2]

                if step % print_freq == 0 and step > 0:
                    # Print training
                    train_acc_ /= print_freq
                    xent_loss_ /= print_freq
                    reg_loss_ /= print_freq
                    print("Step: {}, accuracy: {}, xent loss: {}, reg loss: {}".format(step, train_acc_, xent_loss_,
                                                                                       reg_loss_))

                    # Validation
                    if val_num:
                        ret = sess.run([train_score, train_label, val_score, val_label],
                                       feed_dict={batch_size: 0})
                        train_score_, train_label_, val_score_, val_label_ = ret
                        tprs, fprs, recalls, precisions, val_acc = evaluate(train_score_, train_label_,
                                                                            val_score_, val_label_)
                        print("Validation accuracy: {}".format(val_acc))

                    # Test
                    ret = sess.run([train_score, train_label, test_score, test_label],
                                   feed_dict={batch_size: 0})
                    train_score_, train_label_, test_score_, test_label_ = ret
                    tprs, fprs, recalls, precisions, acc = evaluate(train_score_, train_label_,
                                                                    test_score_, test_label_)
                    print("Test accuracy: {}".format(acc))

                    # write results
                    dur = time.time() - start
                    result_path = os.path.join('result', result_name + '_{}step.pickle'.format(step))
                    with open(result_path, 'wb') as f:
                        pickle.dump({'tprs': tprs,
                                     'fprs': fprs,
                                     'recalls': recalls,
                                     'precisions': precisions,
                                     'accuracy': acc,
                                     'time': dur,
                                     'val_acc': val_acc,
                                     'loss': xent_loss_}, f)
                    print("Time: {} s, accuracy: {}%".format(dur, acc))
