import csv
import os
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt

ROUND_VALUE = 2


def save_fold(cfg, classifier_name, dataset, labels, list_result_fold, list_time, path):
    list_files = []
    for fold in range(0, cfg['fold']):
        # list_fold = list(filter(lambda x, fold=fold: x['fold'] == fold, list_result_fold))
        list_fold = [x for x in list_result_fold if x['fold'] == fold]
        time_fold = [x for x in list_time if x['fold'] == fold]
        # time_fold = list(filter(lambda x, fold=fold: x['fold'] == fold, list_time))

        path_fold = os.path.join(path, str(fold))
        pathlib.Path(path_fold).mkdir(parents=True, exist_ok=True)

        confusion_matrix_by_fold(classifier_name, dataset, labels, list_fold, path_fold)

        index, values = get_values_by_fold_and_metric(list_fold, 'accuracy')
        list_files.append({'filename': 'accuracy', 'index': index, 'path': path_fold, 'values': values})

        index, values = get_values_by_fold_and_metric(list_fold, 'f1_score')
        list_files.append({'filename': 'f1', 'index': index, 'path': path_fold, 'values': values})

        get_top_k_by_rule(list_fold, path_fold)

        index, values = info_by_fold(list_fold, time_fold)
        list_files.append({'filename': 'info_by_fold', 'index': index, 'path': path_fold, 'values': values})

    return list_files


def info_by_fold(list_fold, time):
    index = ['best_rule_accuracy', 'best_accuracy',
             'best_rule_f1', 'best_f1',
             # 'best_rule_top_k', 'best_top_k',
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


def get_top_k_by_rule(list_fold, path_fold):
    for rule in ['max', 'prod', 'sum']:
        # result = list(filter(lambda x, rule=rule: x['rule'] == rule, list_fold))
        result = [x for x in list_fold if x['rule'] == rule]
        if len(result) > 0:
            top_k = result[0]['top_k']
            fold = result[0]['fold']
            max_top_k = result[0]['max_top_k']
            min_top_k = result[0]['min_top_k']
            y_test = result[0]['y_true']
            index = ['rule', 'min_top_k', 'max_top_k', 'total']
            values = [rule, min_top_k, max_top_k, len(y_test)]
            df = pd.DataFrame(top_k)
            df_info = pd.DataFrame(values, index)

            path_top_k_rule = os.path.join(path_fold, 'top_k', rule)
            pathlib.Path(path_top_k_rule).mkdir(exist_ok=True, parents=True)
            save_plot_top_k(fold, max_top_k, min_top_k, path_top_k_rule, rule, top_k, y_test)

            df.to_csv(os.path.join(path_top_k_rule, f'top_k_{rule}.csv'), na_rep='', sep=';', index=False,
                  quoting=csv.QUOTE_ALL)
            df_info.to_csv(os.path.join(path_top_k_rule, f'info_top_k_{rule}.csv'), na_rep='', sep=';', header=False,
                  quoting=csv.QUOTE_ALL)

            path_top_k_rule_xlsx = os.path.join(path_top_k_rule, 'xlsx')
            pathlib.Path(path_top_k_rule_xlsx).mkdir(exist_ok=True, parents=True)
            df.to_excel(os.path.join(path_top_k_rule_xlsx, f'top_k_{rule}.xlsx'), na_rep='', engine='xlsxwriter',
                      index=False)
            df_info.to_excel(os.path.join(path_top_k_rule_xlsx, f'info_top_k_{rule}.xlsx'), na_rep='',
                           engine='xlsxwriter', header=False)


def save_plot_top_k(fold, max_top_k, min_top_k, path_fold, rule, top_k, y_true):
    for k in [3, 5]:
        result_top_k = list(filter(lambda x, k=k: x['k'] <= k, top_k))
        filename = os.path.join(path_fold, f'top_k_{rule}_k={k}.png')
        plot_top_k(filename, fold, k, result_top_k, max_top_k, min_top_k, rule, y_true)

    filename = os.path.join(path_fold, f'top_k_{rule}.png')
    plot_top_k(filename, fold, 'Todos', top_k, max_top_k, min_top_k, rule, y_true)


def plot_top_k(filename, fold, k, list_top_k, max_top_k, min_top_k, rule, y_test):
    x = [top_k['k'] for top_k in list_top_k]
    y = [top_k['top_k_accuracy'] for top_k in list_top_k]
    # :
    #     x.append(top_k['k'])
    #     y.append(top_k['top_k_accuracy'])

    fontsize_title = 14
    pad_title = 20
    fontsize_label = 14

    plt.plot(x, y, marker='o', color='green')
    plt.title(f'Top $k$ accuracy, Rule: {rule}, $k$: {k}, Fold: {fold},\n Min. top $k$: {min_top_k}, Máx. top $k$: {max_top_k}, Número de testes: {len(y_test)}',
        fontsize=fontsize_title, pad=pad_title)
    plt.xlabel('$k$', fontsize=fontsize_label)
    plt.ylabel('Número de acertos', fontsize=fontsize_label)
    plt.grid(True)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.25)
    cfg_plot(filename, plt)


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


def confusion_matrix_by_fold(classifier_name, dataset, list_labels, list_fold, path_fold):
    for rule in ['max', 'prod', 'sum']:
        result = [x for x in list_fold if x['rule'] == rule]
        if len(result) > 0:
            save_confusion_matrix(classifier_name, dataset, list_labels, path_fold, result[0])


def save_confusion_matrix(classifier_name, dataset, labels, path, result):
    rule = result['rule']
    filename = f'ConfusionMatrix_{rule}.png'
    confusion_matrix = ConfusionMatrixDisplay(result['confusion_matrix'], display_labels=labels)

    title = f'Confusion Matrix\nDataset: {dataset}, Classifier: {classifier_name}\nRule: {rule}'
    fontsize_title = 18
    pad_title = 20
    fontsize_labels = 14
    if len(labels) > 5:
        rotation = 90
    else:
        rotation = 45
    plot_size = (10, 10)

    figure, axis = plt.subplots(figsize=plot_size)
    confusion_matrix.plot(ax=axis, cmap='Reds')
    axis.set_title(title, fontsize=fontsize_title, pad=pad_title)
    axis.set_xlabel('y_true', fontsize=fontsize_labels)
    axis.set_ylabel('y_pred', fontsize=fontsize_labels)

    plt.xticks(np.arange(len(labels)), rotation=rotation, fontsize=fontsize_labels)
    plt.yticks(np.arange(len(labels)), fontsize=fontsize_labels)

    plt.gcf().subplots_adjust(bottom=0.15, left=0.25)

    path_confusion_matrix = os.path.join(path, 'confusion_matrix')
    pathlib.Path(path_confusion_matrix).mkdir(exist_ok=True, parents=True)

    cfg_plot(os.path.join(path_confusion_matrix, filename), plt)


def cfg_plot(filename, plt):
    plt.ioff()
    plt.rcParams['figure.facecolor'] = 'white'
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.cla()
    plt.clf()
    plt.close()


def italic_string_plot(string):
    return f'$\\it{{{string}}}$'


def get_list_label(filename):
    with open(filename) as file:
        lines = file.readlines()
        file.close()

        lines = [l for l in lines if len(l) > 0]
        return [italic_string_plot(l.split('->')[1].replace('\n', '')) for l in lines if len(l.split('->')) > 0]