import csv
import os

import numpy as np
import pathlib
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

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

        confusion_matrix_by_fold(classifier_name, data, labels, list_fold, path_fold)

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


def get_only_labels(list_labels):
    return [label['taxon_italic'] for label in list_labels]


def get_labels_and_count_samples(list_labels, list_samples_per_label, n_patch):
    return [label['taxon_italic'] + ' (' + str(int(list_samples_per_label[label['id']] / n_patch)) + ')'
            for label in list_labels]


def save_confusion_matrix_normalized(confusion_matrix, path, rule, xticklabels, yticklabels):
    filename = os.path.join(path, f'ConfusionMatrix_{rule}_normalized.png')
    print(f'save {filename}')
    save_confusion_matrix(confusion_matrix, filename, 'Confusion Matrix', figsize=(35, 35), fmt='.2f',
                          xticklabels=xticklabels, yticklabels=yticklabels, rotation_xtickslabels=90,
                          rotation_ytickslabels=0)


def save_confusion_matrix(confusion_matrix, filename, title, figsize=(5, 5), fmt='.2g', xticklabels=None, yticklabels=None, rotation_xtickslabels=0, rotation_ytickslabels=90):
    vmin = np.min(confusion_matrix)
    vmax = np.max(confusion_matrix)
    off_diag_mask = np.eye(*confusion_matrix.shape, dtype=bool)

    figure, axis = plt.subplots(figsize=figsize)
    axis = sns.heatmap(confusion_matrix, annot=True, mask=~off_diag_mask, cmap='Reds', fmt=fmt, vmin=vmin, vmax=vmax, ax=axis, annot_kws={'fontweight':'bold', 'size': 12})
    axis = sns.heatmap(confusion_matrix, annot=True, mask=off_diag_mask, cmap='Reds', fmt=fmt, vmin=vmin, vmax=vmax, cbar=False, ax=axis)

    fontsize_ticklabels = 8
    axis.set_xticklabels(xticklabels, fontsize=fontsize_ticklabels, rotation=rotation_xtickslabels)
    axis.set_yticklabels(yticklabels, fontsize=fontsize_ticklabels, rotation=rotation_ytickslabels)
    axis.set_xlabel('True label', fontsize=14)
    axis.set_ylabel('Prediction label', fontsize=14)
    axis.set_facecolor('white')
    axis.set_title(title, fontsize=24, pad=32)

    plt.ioff()
    plt.tight_layout()
    plt.savefig(filename, format='png')
    plt.cla()
    plt.clf()


def italic_string_plot(string):
    string = string.replace('\"', '')
    return f'$\\it{{{string}}}$'


def get_string_confusion_matrix(string):
    string = string.replace('\n', '')
    string = string.replace('\"', '')
    taxon_italic = italic_string_plot(string.split(';')[0])
    taxon = string.split(';')[0]
    id = string.split(';')[1]
    id = int(id.replace('f', ''))
    count = string.split(';')[2]
    return {'taxon_italic': taxon_italic, 'taxon': taxon, 'id': id, 'count': count}


def get_list_label(filename):
    try:
        with open(filename) as file:
            lines = file.readlines()
            file.close()

        lines = [l for l in lines if len(l) > 0]
        return [get_string_confusion_matrix(l) for l in lines if len(l.split(';')) > 0]
    except FileNotFoundError:
        print(f'{filename} not exits exists')