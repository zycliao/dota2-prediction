from preprocessing import read_data, onehot_encode
from sklearn import svm
from evaluate import evaluate
import numpy as np

if __name__ == '__main__':
    train_file = 'dota2Train.csv'
    test_file = 'dota2Test.csv'
    train_data, train_label = read_data(train_file)
    test_data, test_label = read_data(test_file)
    train_data = train_data[:5000, :]
    train_label = train_label[:5000]
    test_data = test_data[:1000, :]
    test_label = test_label[:1000]

    train_data = onehot_encode([train_data])[0]
    test_data = onehot_encode([test_data])[0]

    classifier = svm.SVC(probability=True)
    classifier.fit(train_data, train_label)
    train_score = classifier.predict_log_proba(train_data)
    test_score = classifier.predict_log_proba(test_data)
    train_score = np.exp(train_score)
    test_score = np.exp(test_score)
    tprs, fprs, recalls, precisions, acc = evaluate(train_score, train_label, test_score, test_label)
    print acc
