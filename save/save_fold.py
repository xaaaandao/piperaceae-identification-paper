import os

import pathlib
import pandas as pd

from save.save_confusion_matrix import save_confusion_matrix_fold
from save.save_files import save_df
from save.save_top_k import save_top_k_rule

ROUND_VALUE = 2


def save_classification_report(list_fold, path):
    p = os.path.join(path, 'result_per_label')
    pathlib.Path(p).mkdir(exist_ok=True, parents=True)

    for fold in list_fold:
        df = pd.DataFrame(fold['classification_report'])
        df = df.transpose()
        filename = 'result_per_label=%s' % fold['rule']
        save_df(df, filename, p)


def save_fold(cfg, data, list_labels, list_result_fold, list_time, path):
    for fold in range(0, cfg['fold']):
        list_fold = [x for x in list_result_fold if x['fold'] == fold]
        time_fold = [x for x in list_time if x['fold'] == fold]

        path_fold = os.path.join(path, str(fold))
        pathlib.Path(path_fold).mkdir(parents=True, exist_ok=True)

        index_accuracy = []
        values_accuracy = []
        index_f1 = []
        values_f1 = []

        for rule in ['max', 'prod', 'sum']:
            result = [x for x in list_fold if x['rule'] == rule]
            if len(result) > 0:
                save_confusion_matrix_fold(data, list_labels, path_fold, result, rule)

                save_values_of_metric_by_fold(index_accuracy, 'accuracy', result, rule, values_accuracy)
                save_values_of_metric_by_fold(index_f1, 'f1_score', result, rule, values_f1)

                save_top_k_rule(path_fold, result, rule)
                save_classification_report(list_fold, path_fold)

        df = pd.DataFrame(values_accuracy, index_accuracy)
        save_df(df, 'accuracy', path_fold, header=False)
        df = pd.DataFrame(values_f1, index_f1)
        save_df(df, 'f1', path_fold, header=False)

        index_info_fold, values_info_fold = info_fold(list_fold, time_fold)
        df = pd.DataFrame(values_info_fold, index_info_fold)
        save_df(df, 'info_fold', path_fold, header=False)


def info_fold(list_fold, time):
    index = get_index_sheet_fold()
    best_rule_accuracy = max(list_fold, key=lambda x: x['accuracy'])
    best_rule_f1 = max(list_fold, key=lambda x: x['f1_score'])

    time_train_valid = time[0]['time_train_valid']
    time_search_best_params = time[0]['time_search_best_params']

    values = get_values_sheet_fold(best_rule_accuracy, best_rule_f1, time_search_best_params, time_train_valid)
    return index, values


def get_values_sheet_fold(best_rule_accuracy, best_rule_f1, time_search_best_params, time_train_valid):
    return [
        [best_rule_accuracy['rule']],
        [best_rule_accuracy['accuracy'], round(best_rule_accuracy['accuracy'] * 100, ROUND_VALUE)],
        [best_rule_f1['rule']], [best_rule_f1['f1_score'], round(best_rule_f1['f1_score'] * 100, ROUND_VALUE)],
        [time_train_valid, round(time_train_valid, ROUND_VALUE)],
        [time_search_best_params, round(time_search_best_params, ROUND_VALUE)]
    ]


def get_index_sheet_fold():
    return ['best_rule_accuracy', 'best_accuracy',
            'best_rule_f1', 'best_f1',
            'time_train_valid', 'time_search_best_params']


def save_values_of_metric_by_fold(index, metric, result, rule, values):
    index.append(rule)
    value_metric = result[0][metric]
    round_value_metric = round(result[0][metric], ROUND_VALUE)
    values.append([value_metric, round_value_metric])