import os
import pathlib

import numpy as np
import pandas as pd
import sklearn.metrics
from matplotlib import pyplot as plt

ROUND_VALUE = 2


def save_fold(cfg, classifier_name, dataset, list_result_fold, list_time, path):
    list_files = []
    for fold in range(0, cfg['fold']):
        list_fold = list(filter(lambda x: x['fold'] == fold, list_result_fold))
        # print(len(list_fold))
        time_fold = list(filter(lambda x: x['fold'] == fold, list_time))

        path_fold = os.path.join(path, str(fold))
        pathlib.Path(path_fold).mkdir(parents=True, exist_ok=True)

        # confusion_matrix_by_fold(classifier_name, dataset, list_fold, path_fold)

        index, values = get_values_by_fold_and_metric(list_fold, 'accuracy')
        list_files.append({'filename': 'accuracy', 'index': index, 'path': path_fold, 'values': values})
        # create_file_xlsx_and_csv('accuracy', index, path_fold, values)

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
    # best_rule_top_k = max(list_fold, key=lambda x: x['top_k'])
    time_train_valid = time[0]['time_train_valid']
    time_search_best_params = time[0]['time_search_best_params']

    values = [
        [best_rule_accuracy['rule']],
        [best_rule_accuracy['accuracy'], round(best_rule_accuracy['accuracy'] * 100, ROUND_VALUE)],
        [best_rule_f1['rule']], [best_rule_f1['f1_score'], round(best_rule_f1['f1_score'] * 100, ROUND_VALUE)],
        # [best_rule_top_k['rule']], [best_rule_top_k['top_k'], round(best_rule_top_k['top_k'] * 100, ROUND_VALUE)],
        [time_train_valid, round(time_train_valid, ROUND_VALUE)],
        [time_search_best_params, round(time_search_best_params, ROUND_VALUE)]
    ]
    return index, values


def get_top_k_by_rule(list_fold, path_fold):
    for rule in ['max', 'prod', 'sum']:
        result = list(filter(lambda x: x['rule'] == rule, list_fold))
        if len(result) > 0:
            top_k = result[0]['top_k']
            df = pd.DataFrame(top_k)
            df.to_excel(os.path.join(path_fold, f'top_k_{rule}.xlsx'), na_rep='', engine='xlsxwriter', index=False)


def get_values_by_fold_and_metric(list_fold, metric):
    index = []
    values = []
    for rule in ['max', 'prod', 'sum']:
        result = list(filter(lambda x: x['rule'] == rule, list_fold))
        if len(result) > 0:
            index.append(rule)
            value_metric = result[0][metric]
            round_value_metric = round(result[0][metric], ROUND_VALUE)
            values.append([value_metric, round_value_metric])
    return index, values


def confusion_matrix_by_fold(classifier_name, dataset, list_fold, path_fold):
    for rule in ['max', 'prod', 'sum']:
        result = list(filter(lambda x: x['rule'] == rule, list_fold))
        save_confusion_matrix(classifier_name, dataset, path_fold, result[0])


def save_confusion_matrix(classifier_name, dataset, path, result):
    filename = f'confusion_matrix_{result["rule"]}.png'
    # cinco labels -> IWSSIP
    labels = ['$\it{Manekia}$', '$\it{Ottonia}$', '$\it{Peperomia}$', '$\it{Piper}$', '$\it{Pothomorphe}$']

    # acima de cinco labels -> dataset George
    # labels =

    # acima de cinco dez -> dataset George
    # labels =

    # acima de cinco vinte -> dataset George
    # labels =

    # duas labels -> dataset George
    # labels = ['$\it{Peperomia}$', '$\it{Piper}$']

    confusion_matrix = sklearn.metrics.ConfusionMatrixDisplay(result['confusion_matrix'])
    confusion_matrix.plot(cmap='Reds')

    title = f'Confusion Matrix\ndataset: {dataset}, classifier: {classifier_name}\naccuracy: {round(result["accuracy"], ROUND_VALUE)}, rule: {result["rule"]}'
    fontsize_title = 12
    pad_title = 20

    fontsize_labels = 8

    background_color = 'white'
    plt.ioff()
    plt.title(title, fontsize=fontsize_title, pad=pad_title)
    plt.xticks(np.arange(len(labels)), labels, rotation=45, fontsize=fontsize_labels)
    plt.yticks(np.arange(len(labels)), labels, fontsize=fontsize_labels)
    plt.ylabel('y_test', fontsize=fontsize_labels)
    plt.xlabel('y_pred', fontsize=fontsize_labels)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.25)
    plt.rcParams['figure.facecolor'] = background_color
    plt.tight_layout()
    plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()
