from preprocessing import read_data, onehot_encode, data_aug_np
from sklearn import svm
from evaluate import evaluate
import pickle
import os
import time


if __name__ == '__main__':
    result_name = 'svm_onlyheroes' + '.pickle'
    mode = 'one_hot_all_feat'
    train_file = 'dota2Train.csv'
    test_file = 'dota2Test.csv'

    assert os.path.exists('result')

    train_data, train_label = read_data(train_file)
    test_data, test_label = read_data(test_file)
    train_data = train_data[:, :]
    train_label = train_label[:]
    test_data = test_data[:, :]
    test_label = test_label[:]
    test_data, test_label = data_aug_np(test_data, test_label)

    if mode == 'only_heroes':
        train_data = train_data[:, 3:]
        test_data = test_data[:, 3:]
    elif mode == 'all_feat':
        pass
    elif mode == 'one_hot_all_feat':
        train_data, test_data = onehot_encode(train_data, test_data)

    start = time.time()
    classifier = svm.SVC(probability=True)
    classifier.fit(train_data, train_label)
    train_score = classifier.predict_proba(train_data)
    test_score = classifier.predict_proba(test_data)
    tprs, fprs, recalls, precisions, acc = evaluate(train_score, train_label, test_score, test_label)
    dur = time.time() - start

    result_path = os.path.join('result', result_name)
    with open(result_path, 'wb') as f:
        pickle.dump({'tprs': tprs,
                     'fprs': fprs,
                     'recalls': recalls,
                     'precisions': precisions,
                     'accuracy': acc,
                     'time': dur}, f)
    print("Time: {} s, accuracy: {}%".format(dur, acc))
