import collections
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, top_k_accuracy_score


def get_index_max_value(y):
    index = np.unravel_index(np.argmax(y, axis=None), y.shape)  # index return a tuple
    return int(index[1])


def max_rule(n_labels, n_patch, y_pred):
    new_y_pred = np.empty(shape=(0,))
    new_y_pred_prob = np.empty(shape=(0, n_labels))
    for i, j in next_sequence(0, y_pred.shape[0], n_patch):
        new_y_pred = np.append(new_y_pred, get_index_max_value(y_pred[i:j]) + 1)
        new_y_pred_prob = np.vstack((new_y_pred_prob, np.amax(y_pred[i:j], axis=0)))
    return new_y_pred_prob, new_y_pred


def next_sequence(start, end, step):
    for i in range(start, end, step):
        yield i, i + step


def y_test_with_patch(n_patch, y_test):
    new_y_test = np.empty(shape=(0,))
    # for i, j in next_sequence(0, y_test.shape[0], n_patch):
    #     new_y_test = np.append(new_y_test, y_test[i])
    # return new_y_test
    return np.append(new_y_test, [y_test[i] for i, j in next_sequence(0, y_test.shape[0], n_patch)])


def prod_all_prob(n_labels, n_patch, y_pred):
    new_y_pred = np.empty(shape=(0,))
    new_y_pred_prob_prod = np.empty(shape=(0, n_labels))
    for i, j in next_sequence(0, y_pred.shape[0], n_patch):
        new_y_pred = np.append(new_y_pred, np.argmax(y_pred[i:j].prod(axis=0)) + 1)
        new_y_pred_prob_prod = np.vstack((new_y_pred_prob_prod, y_pred[i:j].prod(axis=0)))
    return new_y_pred_prob_prod, new_y_pred


def sum_all_prob(n_labels, n_patch, y_pred):
    new_y_pred = np.empty(shape=(0,))
    new_y_pred_prob_sum = np.empty(shape=(0, n_labels))
    for i, j in next_sequence(0, y_pred.shape[0], n_patch):
        new_y_pred = np.append(new_y_pred, np.argmax(y_pred[i:j].sum(axis=0)) + 1)
        new_y_pred_prob_sum = np.vstack((new_y_pred_prob_sum, y_pred[i:j].sum(axis=0)))
    return new_y_pred_prob_sum, new_y_pred


def calculate_test(fold, n_labels, y_pred, y_test, n_patch=1):
    if n_patch > 1:
        y_test = y_test_with_patch(n_patch, y_test)
    y_pred_prob_max, y_pred_max = max_rule(n_labels, n_patch, y_pred)
    y_pred_prob_prod, y_pred_prod = prod_all_prob(n_labels, n_patch, y_pred)
    y_pred_prob_sum, y_pred_sum = sum_all_prob(n_labels, n_patch, y_pred)
    max = create_result(fold, n_labels, 'max', y_pred_prob_max, y_pred_max, y_test)
    prod = create_result(fold, n_labels, 'prod', y_pred_prob_prod, y_pred_prod, y_test)
    sum = create_result(fold, n_labels, 'sum', y_pred_prob_sum, y_pred_sum, y_test)
    return max, prod, sum


def create_result(fold, n_labels, rule, y_pred_prob, y_pred, y_test):
    accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
    confusion_matrix = confusion_matrix(y_pred=y_pred, y_true=y_test)

    f1_score = 0
    if min(list(collections.Counter(y_test).values())) != max(list(collections.Counter(y_test).values())):
        f1_score = f1_score(y_pred=y_pred, y_true=y_test, average='weighted')

    list_top_k_accuracy = []
    if n_labels > 2:
        list_top_k_accuracy = [{'k': k,
                                'top_k_accuracy': top_k_accuracy_score(y_true=y_test,
                                                                       y_score=y_pred_prob,
                                                                       normalize=False,
                                                                       k=k, labels=np.arange(1, n_labels + 1))}
                                                                        for k in range(3, n_labels)]
    return {
        'fold': fold,
        'rule': rule,
        'y_pred_prob': y_pred_prob,
        'y_pred': y_pred,
        'y_true': y_test,
        'accuracy': accuracy,
        'f1_score': f1_score,
        'top_k': list_top_k_accuracy,
        'max_top_k': max(list_top_k_accuracy, key=lambda x: x['top_k_accuracy'])['top_k_accuracy'] if len(list_top_k_accuracy) > 0 else 0,
        'min_top_k': min(list_top_k_accuracy, key=lambda x: x['top_k_accuracy'])['top_k_accuracy'] if len(list_top_k_accuracy) > 0 else 0,
        'confusion_matrix': confusion_matrix
    }

