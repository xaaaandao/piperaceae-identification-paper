import csv
import io
import os
import multiprocessing

import matplotlib
import numpy as np
import pathlib
import pandas as pd
import PIL

from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.utils import parallel_backend
from matplotlib import pyplot as plt

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


def save_fold(cfg, classifier_name, dataset, labels, list_result_fold, list_time, path):
    list_files = []
    for fold in range(0, cfg['fold']):
        list_fold = [x for x in list_result_fold if x['fold'] == fold]
        time_fold = [x for x in list_time if x['fold'] == fold]

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


def confusion_matrix_by_fold(classifier_name, dataset, list_labels, list_fold, path_fold):
    for rule in ['max', 'prod', 'sum']:
        result = [x for x in list_fold if x['rule'] == rule]
        if len(result) > 0:
            # print(f'plot confusion matrix {result[0]["fold"]} {result[0]["rule"]}')
            path_confusion_matrix = os.path.join(path_fold, 'confusion_matrix', rule)
            pathlib.Path(path_confusion_matrix).mkdir(exist_ok=True, parents=True)

            confusion_matrix = result[0]['confusion_matrix']
            filename = os.path.join(path_confusion_matrix, f'ConfusionMatrix_{rule}.png')
            print(f'save {filename}')
            save_confusion_matrix(classifier_name, confusion_matrix, dataset, filename, 18, list_labels, (15, 15), result[0]['rule'])

            confusion_matrix = result[0]['confusion_matrix_normalized']
            filename = os.path.join(path_confusion_matrix, f'ConfusionMatrix_{rule}_normalized.png')
            print(f'save {filename}')
            save_confusion_matrix(classifier_name, confusion_matrix, dataset, filename, 44, list_labels, (35, 35), result[0]['rule'])


def save_confusion_matrix(classifier_name, confusion_matrix, dataset, filename, fontsize_title, labels, plot_size, rule):
    confusion_matrix = ConfusionMatrixDisplay(confusion_matrix, display_labels=labels)

    title = f'Confusion Matrix\nDataset: {dataset}, Classifier: {classifier_name}\nRule: {rule}'
    color_map = 'Reds'
    pad_title = 20
    fontsize_labels = 14

    rotation = get_rotation(labels)

    figure, axis = plt.subplots(figsize=plot_size)
    confusion_matrix.plot(ax=axis, cmap=color_map)
    axis.set_title(title, fontsize=fontsize_title, pad=pad_title)
    axis.set_xlabel('y_true', fontsize=fontsize_labels)
    axis.set_ylabel('y_pred', fontsize=fontsize_labels)

    plt.xticks(np.arange(len(labels)), rotation=rotation, fontsize=fontsize_labels)
    plt.yticks(np.arange(len(labels)), fontsize=fontsize_labels)

    plt.gcf().subplots_adjust(bottom=0.15, left=0.25)
    plt.ioff()
    plt.rcParams['figure.facecolor'] = 'white'

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', format='png')
    plt.cla()
    plt.clf()
    plt.close()


def get_rotation(labels):
    return 90 if len(labels) > 5 else 45


def italic_string_plot(string):
    string = string.replace("\"", "")
    return f'$\\it{{{string}}}$'


def get_list_label(filename):
    try:
        with open(filename) as file:
            lines = file.readlines()
            file.close()

        lines = [l for l in lines if len(l) > 0]
        return [italic_string_plot(l.split(';')[0].replace('\n', '')) for l in lines if len(l.split(';')) > 0]
    except FileNotFoundError:
        print(f'{filename} not exits exists')