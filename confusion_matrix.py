import collections
import csv
import os
import pathlib

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def save_confusion_matrix_sheet(confusion_matrix, filename, xticklabels, yticklabels):
    index = [label.replace('$\it{', '').replace('}$', '') for label in yticklabels]
    columns = [label.replace('$\it{', '').replace('}$', '') for label in xticklabels]
    columns.append('Threshold')
    l = []
    for i, cm in enumerate(confusion_matrix):
        x = cm.tolist()
        if float(cm[i]) > 0.6:
            x.append('easy')
        elif float(cm[i]) <= 0.6 and float(cm[i]) >= 0.4:
            x.append('medium')
        else:
            x.append('hard')
        l.append(x)

    df = pd.DataFrame(l, index=index, columns=columns)
    df.to_csv(filename + '.csv', sep=';', na_rep='', quoting=csv.QUOTE_ALL)
    df.to_excel(filename + '.xlsx', na_rep='', engine='xlsxwriter')


def confusion_matrix_by_fold(data, list_labels, list_fold, path_fold):
    for rule in ['max', 'prod', 'sum']:
        result = [x for x in list_fold if x['rule'] == rule]
        if len(result) > 0:
            path_confusion_matrix = os.path.join(path_fold, 'confusion_matrix', rule)
            pathlib.Path(path_confusion_matrix).mkdir(exist_ok=True, parents=True)

            list_samples_per_label = dict(collections.Counter(result[0]['y_true']))
            yticklabels = get_labels_and_count_samples(list_labels, list_samples_per_label, data['n_patch'])
            xticklabels = get_only_labels(list_labels)

            save_confusion_matrix_normal(result[0]['confusion_matrix'], path_confusion_matrix, rule, xticklabels, yticklabels)
            save_confusion_matrix_normalized(result[0]['confusion_matrix_normalized'], path_confusion_matrix, rule, xticklabels, yticklabels)
            save_confusion_matrix_sheet(result[0]['confusion_matrix'], os.path.join(path_confusion_matrix, f'confusionmatrix_normalized_{rule}'), xticklabels, yticklabels)


def save_confusion_matrix_normal(confusion_matrix, path_confusion_matrix, rule, xticklabels, yticklabels):
    filename = os.path.join(path_confusion_matrix, f'ConfusionMatrix_{rule}.png')
    print(f'save {filename}')
    save_confusion_matrix(confusion_matrix, filename, 'Confusion Matrix', figsize=(5, 5), fmt='.2g', xticklabels=xticklabels, yticklabels=yticklabels, rotation_xtickslabels=90, rotation_ytickslabels=0)


def get_only_labels(list_labels):
    return [label['taxon_italic'] for label in list_labels]


def get_labels_and_count_samples(list_labels, list_samples_per_label, n_patch):
    return [label['taxon_italic'] + ' (' + str(int(list_samples_per_label[label['id']] / n_patch)) + ')'
            for label in list_labels]


def save_confusion_matrix_normalized(confusion_matrix, path, rule, xticklabels, yticklabels):
    filename = os.path.join(path, f'ConfusionMatrix_{rule}_normalized.png')
    print(f'[CONFUSION MATRIX] save {filename}')
    save_confusion_matrix(confusion_matrix, filename, 'Confusion Matrix', figsize=(5, 5), fmt='.2f',
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
    plt.close(figure)


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
