import numpy as np


def calc_metrics(thresh, predict_score, label):
    predict = predict_score[:, 1] > thresh
    tp = np.sum(np.logical_and(label, predict), dtype=np.float32)
    fp = np.sum(np.logical_and(np.logical_not(label), predict), dtype=np.float32)
    tn = np.sum(np.logical_and(np.logical_not(label), np.logical_not(predict)), dtype=np.float32)
    fn = np.sum(np.logical_and(label, np.logical_not(predict)), dtype=np.float32)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    acc = (tp + tn) / (tp + fp + tn + fn)
    return tpr, fpr, recall, precision, acc


def evaluate(train_score, train_label, val_score, val_label):
    """
    Evaluate tpr, fpr, recall, precision, accuracy
    :param train_score: shape: [sample_num, 2]
    :param train_label: shape: [sample_num, ]
    :param val_score: shape: [sample_num, 2]
    :param val_label: shape: [sample_num, ]
    :return: the shape of tpr, fpr, recall, precision is [threshold_num,],
    accuracy is a scalar
    """
    thresholds = np.arange(0, 1, 0.01)
    thresh_num = thresholds.shape[0]
    tprs = np.zeros([thresh_num, ])
    fprs = np.zeros([thresh_num, ])
    recalls = np.zeros([thresh_num, ])
    precisions = np.zeros([thresh_num, ])

    max_acc = 0.
    optim_thresh = 0.
    for thresh_idx in range(thresh_num):
        thresh = thresholds[thresh_idx]
        _, _, _, _, acc = calc_metrics(thresh, train_score, train_label)
        if acc >= max_acc:
            max_acc = acc
            optim_thresh = thresh

    for thresh_idx in range(thresh_num):
        thresh = thresholds[thresh_idx]
        tpr, fpr, recall, precision, _ = calc_metrics(thresh, val_score, val_label)
        tprs[thresh_idx] = tpr
        fprs[thresh_idx] = fpr
        recalls[thresh_idx] = recall
        precisions[thresh_idx] = precision

    _, _, _, _, acc = calc_metrics(optim_thresh, val_score, val_label)
    return tprs, fprs, recalls, precisions, acc
