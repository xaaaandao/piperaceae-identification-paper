import collections
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, top_k_accuracy_score, \
    multilabel_confusion_matrix


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


def create_result(fold, n_labels, rule, y_pred_prob, y_pred, y_true):
    accuracy = accuracy_score(y_pred=y_pred, y_true=y_true)
    cm = confusion_matrix(y_pred=y_pred, y_true=y_true)
    cm_normalized = confusion_matrix(y_pred=y_pred, y_true=y_true, normalize='true')
    cm_multilabel = multilabel_confusion_matrix(y_pred=y_pred, y_true=y_true)

    f1 = 0
    if min(list(collections.Counter(y_true).values())) != max(list(collections.Counter(y_true).values())):
        f1 = f1_score(y_pred=y_pred, y_true=y_true, average='weighted')

    list_top_k_accuracy = []
    if n_labels > 2:
        list_top_k_accuracy = [{'k': k,
                                'top_k_accuracy': top_k_accuracy_score(y_true=y_true,
                                                                       y_score=y_pred_prob,
                                                                       normalize=False,
                                                                       k=k, labels=np.arange(1, n_labels + 1))}
                                                                        for k in range(3, n_labels)]

    cr = classification_report(y_pred=y_pred, y_true=y_true, labels=np.arange(1, n_labels + 1), zero_division=0, output_dict=True)
    return {
        'fold': fold,
        'rule': rule,
        'y_pred_prob': y_pred_prob,
        'y_pred': y_pred,
        'y_true': y_true,
        'accuracy': accuracy,
        'f1_score': f1,
        'top_k': list_top_k_accuracy,
        'max_top_k': get_max_top_k(list_top_k_accuracy),
        'min_top_k': get_min_top_k(list_top_k_accuracy),
        'confusion_matrix': cm,
        'confusion_matrix_normalized': cm_normalized,
        'confusion_matrix_multilabel': cm_multilabel,
        'classification_report': cr,
    }

def get_max_top_k(list_top_k_accuracy):
    if len(list_top_k_accuracy) > 0:
        max_top_k = max(list_top_k_accuracy, key=lambda x: x['top_k_accuracy'])
        return max_top_k['top_k_accuracy']
    return 0


def get_min_top_k(list_top_k_accuracy):
    if len(list_top_k_accuracy) > 0:
        min_top_k = min(list_top_k_accuracy, key=lambda x: x['top_k_accuracy'])
        return min_top_k['top_k_accuracy']
    return 0


def insert_result_fold_and_time(end_time_train_valid, fold, list_result_fold, list_time, result_max_rule, result_prod_rule, result_sum_rule,
                                start_time_train_valid, time_find_best_params):
    list_result_fold.append(result_max_rule)
    list_result_fold.append(result_prod_rule)
    list_result_fold.append(result_sum_rule)
    list_time.append({
        'fold': fold,
        'time_train_valid': end_time_train_valid - start_time_train_valid,
        'time_search_best_params': time_find_best_params
    })
