import csv
import itertools
import matplotlib.pyplot as plt
import os
import pathlib

import pandas as pd

import numpy as np

from save_fold import cfg_plot

ROUND_VALUE = 2


def mean_top_k(list_result_fold, path):
    p = os.path.join(path, 'mean_top_k')
    pathlib.Path(p).mkdir(exist_ok=True, parents=True)
    for rule in ['max', 'prod', 'sum']:
        list_top_k = [x['top_k'] for x in list_result_fold if x['rule'] == rule]
        list_top_k = list(itertools.chain.from_iterable(list_top_k))

        min_k = min(list_top_k, key=lambda x: x['k'])['k']
        max_k = max(list_top_k, key=lambda x: x['k'])['k']
        # max =

        list_each_k = []
        for i in range(min_k, max_k + 1):
            values_k = [k['top_k_accuracy'] for k in list_top_k if k['k'] == i]
            list_each_k.append({'k': i, 'values': values_k, 'mean': np.mean(values_k)})

        df = pd.DataFrame(list_each_k)

        filename = f'mean_top_k_{rule}'
        df.to_csv(os.path.join(p, f'{filename}.csv'), sep=';', na_rep='', quoting=csv.QUOTE_ALL, index=False)
        df.to_excel(os.path.join(p, f'{filename}.xlsx'), na_rep='', engine='xlsxwriter', index=False)

        axis_x = [k['k'] for k in list_each_k]
        axis_y = [k['mean'] for k in list_each_k]
        fontsize_title = 14
        pad_title = 20
        fontsize_label = 14

        plt.plot(axis_x, axis_y, marker='o', color='green')
        plt.title(
            'test',
            fontsize=fontsize_title, pad=pad_title)
        plt.xlabel('k', fontsize=fontsize_label)
        plt.ylabel('Número de acertos', fontsize=fontsize_label)
        plt.grid(True)
        plt.gcf().subplots_adjust(bottom=0.15, left=0.25)
        cfg_plot(os.path.join(p, f'{filename}.png'), plt)
        # axis_y =

def save_mean(list_result_fold, list_time, path):
    mean_time_train_valid, std_time_train_valid = get_mean_std_time('time_train_valid', list_time)
    mean_time_search_best_params, std_time_search_best_params = get_mean_std_time('time_search_best_params', list_time)
    # mean

    list_mean_accuracy = get_list_all_rule(list_result_fold, 'accuracy')
    list_mean_f1 = get_list_all_rule(list_result_fold, 'f1_score')

    accuracy = get_mean_metric(list_mean_accuracy)
    f1 = get_mean_metric(list_mean_f1)
    mean_top_k(list_result_fold, path)

    values = [mean_time_train_valid, std_time_train_valid,
              mean_time_search_best_params, std_time_search_best_params] + accuracy + f1

    index = ['mean_time_train_valid', 'std_time_train_valid',
             'mean_time_search_best_params', 'std_time_search_best_params',
             'mean_accuracy_max', 'std_accuracy_max',
             'mean_accuracy_sum', 'std_accuracy_sum',
             'mean_accuracy_prod', 'std_accuracy_prod',
             'mean_f1_max', 'std_f1_max',
             'mean_f1_sum', 'std_f1_sum',
             'mean_f1_prod', 'std_f1_prod']

    return list_mean_accuracy, list_mean_f1, \
           [{'filename': 'mean', 'index': index, 'path': path, 'values': values}]


def get_mean_metric(list_mean_metric):
    # metric_max = list(filter(lambda x: x['rule'] == 'max', list_mean_metric))
    metric_max = [m for m in list_mean_metric if m['rule'] == 'max']
    mean_metric_max = metric_max[0]['mean']
    std_metric_max = metric_max[0]['std']

    # metric_sum = list(filter(lambda x: x['rule'] == 'sum', list_mean_metric))
    metric_sum = [m for m in list_mean_metric if m['rule'] == 'sum']
    mean_metric_sum = metric_sum[0]['mean']
    std_metric_sum = metric_sum[0]['std']

    # metric_prod = list(filter(lambda x: x['rule'] == 'prod', list_mean_metric))
    metric_prod = [m for m in list_mean_metric if m['rule'] == 'prod']
    mean_metric_prod = metric_prod[0]['mean']
    std_metric_prod = metric_prod[0]['std']

    return [[mean_metric_max], [std_metric_max],
            [mean_metric_sum], [std_metric_sum],
            [mean_metric_prod], [std_metric_prod]]


def get_list_all_rule(list_result_fold, metric):
    return [calculate_mean_by_metric_and_rule(list_result_fold, metric, 'sum'),
            calculate_mean_by_metric_and_rule(list_result_fold, metric, 'max'),
            calculate_mean_by_metric_and_rule(list_result_fold, metric, 'prod')]


def calculate_mean_by_metric_and_rule(list_result_fold, metric, rule):
    list_rule = [x for x in list_result_fold if x['rule'] == rule]
    return {
        'mean': np.mean([r[metric] for r in list_rule]),
        'std': np.std([r[metric] for r in list_rule]),
        'rule': rule,
        'metric': metric
    }


def get_mean_std_time(key, list_time):
    mean_time = np.mean([time[key] for time in list_time])
    round_mean_time = round(float(mean_time), ROUND_VALUE)

    std_time = np.std([time[key] for time in list_time])
    round_std_time = round(float(std_time), ROUND_VALUE)

    return [mean_time, round_mean_time], [std_time, round_std_time]
