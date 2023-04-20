import collections
import os
import timeit

import numpy as np
import logging

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, multilabel_confusion_matrix, \
    classification_report, top_k_accuracy_score

from arrays import split_dataset, mult_rule, max_rule, sum_rule, y_true_no_patch
from save import save_fold, save_confusion_matrix


def run_folds(classifier, classifier_name, fold, index_train, index_test, list_info_level, n_features, patch,
              path, x, y):
    logging.info('[INFO] fold: %d classifier name: %s' % (fold, classifier_name))
    x_train, y_train = split_dataset(index_train, n_features, patch, x, y)
    x_test, y_test = split_dataset(index_test, n_features, patch, x, y)

    count_train = collections.Counter(y_train)
    count_test = collections.Counter(y_test)
    logging.info('[INFO] TRAIN: %s ' % str(sorted(count_train.items())))
    logging.info('[INFO] TEST: %s ' % str(sorted(count_test.items())))

    logging.info('[INFO] x_train.shape: %s y_train.shape: %s' % (str(x_train.shape), str(y_train.shape)))
    logging.info('[INFO] x_test.shape: %s y_test.shape: %s' % (str(x_test.shape), str(y_test.shape)))

    start_timeit = timeit.default_timer()

    classifier.best_estimator_.fit(x_train, y_train)

    y_pred_proba = classifier.best_estimator_.predict_proba(x_test)
    end_timeit = timeit.default_timer() - start_timeit

    n_test, n_labels = y_pred_proba.shape
    y_pred_max_rule, y_score_max = max_rule(n_test, n_labels, patch, y_pred_proba)
    y_pred_mult_rule, y_score_mult = mult_rule(n_test, n_labels, patch, y_pred_proba)
    y_pred_sum_rule, y_score_sum = sum_rule(n_test, n_labels, patch, y_pred_proba)
    y_true = y_true_no_patch(n_test, patch, y_test)

    logging.info('[INFO] y_pred_sum_rule.shape: %s y_score_sum: %s' % (
        str(y_pred_sum_rule.shape), str(y_score_sum.shape)))
    logging.info('[INFO] y_pred_mult_rule.shape: %s y_score_mult: %s' % (
        str(y_pred_mult_rule.shape), str(y_score_mult.shape)))
    logging.info('[INFO] y_pred_max_rule.shape: %s y_score_max: %s' % (
        str(y_pred_max_rule.shape), str(y_score_max.shape)))
    logging.info('[INFO] y_true.shape: %s' % str(y_true.shape))

    results = {
        'fold': fold,
        'max': evaluate(list_info_level, n_labels, y_pred_max_rule, y_score_max, y_true),
        'mult': evaluate(list_info_level, n_labels, y_pred_mult_rule, y_score_mult, y_true),
        'sum': evaluate(list_info_level, n_labels, y_pred_sum_rule, y_score_sum, y_true),
        'time': end_timeit,
        'total_train': np.sum([v for v in count_train.values()]),
        'total_test': np.sum([v for v in count_test.values()]),
        'total_train_no_patch': int(np.sum([v for v in count_train.values()])/patch),
        'total_test_no_patch': int(np.sum([v for v in count_test.values()])/patch)
    }

    path_fold = os.path.join(path, str(fold))

    if not os.path.exists(path_fold):
        os.makedirs(path_fold)

    save_fold(count_train, count_test, path_fold, results)
    save_confusion_matrix(count_train, count_test, list_info_level, patch, path_fold, results)
    return results, n_labels


def evaluate(list_info_level, n_labels, y_pred, y_score, y_true):
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    accuracy = accuracy_score(y_pred=y_pred, y_true=y_true)
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    cm_normalized = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true')
    cm_multilabel = multilabel_confusion_matrix(y_pred=y_pred, y_true=y_true)
    cr = classification_report(y_pred=y_pred, y_true=y_true, labels=np.arange(1, len(list_info_level['levels']) + 1),
                               target_names=['label_%s+%s' % (i, label) for i, label in
                                             enumerate(list_info_level['levels'].values(), start=1)], zero_division=0,
                               output_dict=True)
    list_topk = [
        {'k': k,
         'top_k_accuracy': top_k_accuracy_score(y_true=y_true, y_score=y_score, normalize=False, k=k,
                                                labels=np.arange(1, len(list_info_level['levels']) + 1))}
        for k in range(3, n_labels)
    ]
    return {'f1': f1,
            'confusion_matrix': cm,
            'confusion_matrix_normalized': cm_normalized,
            'confusion_matrix_multilabel': cm_multilabel,
            'classification_report': cr, 'list_topk': list_topk, 'accuracy': accuracy}
