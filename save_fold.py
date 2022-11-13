import csv
import os

import pathlib
import pandas as pd

from confusion_matrix import confusion_matrix_by_fold
from save_top_k import get_top_k_by_rule

ROUND_VALUE = 2


def result_per_label(list_fold, path):
    p = os.path.join(path, 'result_per_label')
    pathlib.Path(p).mkdir(exist_ok=True, parents=True)
    for f in list_fold:
        df = pd.DataFrame(f['classification_report'])
        df = df.transpose()
        filename = f'result_per_label={f["rule"]}'
        df.to_csv(os.path.join(p, f'{filename}.csv'), sep=';', na_rep='', quoting=csv.QUOTE_ALL)
        df.to_excel(os.path.join(p, f'{filename}.xlsx'), na_rep='', engine='xlsxwriter')


def save_fold(cfg, classifier_name, data, labels, list_result_fold, list_time, path):
    list_files = []
    for fold in range(0, cfg['fold']):
        list_fold = [x for x in list_result_fold if x['fold'] == fold]
        time_fold = [x for x in list_time if x['fold'] == fold]

        path_fold = os.path.join(path, str(fold))
        pathlib.Path(path_fold).mkdir(parents=True, exist_ok=True)

        confusion_matrix_by_fold(data, labels, list_fold, path_fold)

        index, values = get_values_by_fold_and_metric(list_fold, 'accuracy')
        list_files.append({'filename': 'accuracy', 'index': index, 'path': path_fold, 'values': values})

        index, values = get_values_by_fold_and_metric(list_fold, 'f1_score')
        list_files.append({'filename': 'f1', 'index': index, 'path': path_fold, 'values': values})

        get_top_k_by_rule(list_fold, path_fold)

        index, values = info_by_fold(list_fold, time_fold)
        list_files.append({'filename': 'info_by_fold', 'index': index, 'path': path_fold, 'values': values})

        result_per_label(list_fold, path_fold)

    return list_files


def info_by_fold(list_fold, time):
    index = ['best_rule_accuracy', 'best_accuracy',
             'best_rule_f1', 'best_f1',
             'time_train_valid', 'time_search_best_params']
    best_rule_accuracy = max(list_fold, key=lambda x: x['accuracy'])
    best_rule_f1 = max(list_fold, key=lambda x: x['f1_score'])

    time_train_valid = time[0]['time_train_valid']
    time_search_best_params = time[0]['time_search_best_params']

    values = [
        [best_rule_accuracy['rule']],
        [best_rule_accuracy['accuracy'], round(best_rule_accuracy['accuracy'] * 100, ROUND_VALUE)],
        [best_rule_f1['rule']], [best_rule_f1['f1_score'], round(best_rule_f1['f1_score'] * 100, ROUND_VALUE)],
        [time_train_valid, round(time_train_valid, ROUND_VALUE)],
        [time_search_best_params, round(time_search_best_params, ROUND_VALUE)]
    ]
    return index, values


def get_values_by_fold_and_metric(list_fold, metric):
    index = []
    values = []
    for rule in ['max', 'prod', 'sum']:
        result = [x for x in list_fold if x['rule'] == rule]
        if len(result) > 0:
            index.append(rule)
            value_metric = result[0][metric]
            round_value_metric = round(result[0][metric], ROUND_VALUE)
            values.append([value_metric, round_value_metric])
    return index, values


