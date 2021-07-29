import numpy as np
from mlthon.mlthon.backend import dim_check, check_data_validity


def _checks(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError("Length of 'y_true' and 'y_pred' should be same.")
    check_data_validity([y_true, y_pred], names=['y_true', 'y_pred'])
    dim_check([y_true, y_pred], [1, 1], ['y_true', 'y_pred'])


def accuracy_score(y_true, y_pred):
    _checks(y_true, y_pred)
    return sum(y_true == y_pred)/len(y_true)


def area_under_curve(X, y):
    return np.trapz(y, X)


# "balanced_accuracy_score"
# "brier_accuracy_score"
# "classification_report"
# "cohen_kappa_score"


def confusion_matrix(y_true, y_pred):
    _checks(y_true, y_pred)

    classes = np.unique(np.concatenate((y_true, y_pred)))
    n = len(classes)
    cm = np.zeros(shape=(n, n), dtype=np.int32)
    for i, j in zip(y_true, y_pred):
        cm[np.where(classes == i)[0], np.where(classes == j)[0]] += 1

    return cm


# "dcg_score"
# "det_score"


def f1_score(y_true, y_pred, average='auto'):
    precision = precision_score(y_true, y_pred, average)
    recall = recall_score(y_true, y_pred, average)
    return 2 * (precision * recall)/(precision + recall)


# "fbeta_score"
# "hamming_loss"
# "hinge_loss"
# "jaccard_score"
# "log_loss"
# "matthews_corrcoef"
# "ndcg_score"
# "precision_recall_curve"


def precision_score(y_true, y_pred, average='auto'):
    _checks(y_true, y_pred)

    classes = np.unique(np.concatenate((y_true, y_pred)))
    if average == 'auto':
        if len(classes) == 2:
            average = 'binary'
        else:
            average = 'micro'

    cm = confusion_matrix(y_true, y_pred)
    if average == 'binary':
        tp, fp = cm.ravel()[:2]
        return tp/(tp + fp)
    
    if average == 'micro':
        tp, fp = list(), list()
        for i in range(len(cm)):
            tp.append(cm[i, i])
            fp.append(sum(np.delete(cm[i], i)))

        tp_all = sum(tp)
        fp_all = sum(fp)
        return tp_all/(tp_all + fp_all)


def recall_score(y_true, y_pred, average='auto'):
    _checks(y_true, y_pred)

    classes = np.unique(np.concatenate((y_true, y_pred)))
    if average == 'auto':
        if len(classes) == 2:
            average = 'binary'
        else:
            average = 'micro'

    cm = confusion_matrix(y_true, y_pred)
    if average == 'binary':
        tp, fn = cm.ravel()[[0, 2]]
        return tp/(tp + fn)

    if average == 'micro':
        tp, fn = list(), list()
        for i in range(len(cm)):
            tp.append(cm[i, i])
            fn.append(sum(np.delete(cm[:, i], i)))

        tp_all = sum(tp)
        fn_all = sum(fn)
        return tp_all/(tp_all + fn_all)
