import collections
import csv
import os
import pathlib
import shutil

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from save.save_files import compress_folder


def save_confusion_matrix_sheet(confusion_matrix, filename, xticklabels, yticklabels):
    index = [label.replace('$\it{', '').replace('}$', '') for label in yticklabels]
    columns = [label.replace('$\it{', '').replace('}$', '') for label in xticklabels]
    columns.append('Threshold')
    l = classify_values_confusion_matrix(confusion_matrix)

    df = pd.DataFrame(l, index=index, columns=columns)
    df.to_csv(filename + '.csv', sep=';', na_rep='', quoting=csv.QUOTE_ALL)
    # df.to_excel(filename + '.xlsx', na_rep='', engine='xlsxwriter')


def classify_values_confusion_matrix(confusion_matrix):
    list_values = []
    for i, cm in enumerate(confusion_matrix):
        line_confusion_matrix = cm.tolist()
        check_accuracy(cm, line_confusion_matrix, i)
        list_values.append(line_confusion_matrix)
    return list_values


def accuracy_greather_than_six(accuracy):
    return float(accuracy) > 0.6


def accuracy_grather_than_four_and_less_than_six(accuracy):
    return 0.6 >= float(accuracy) >= 0.4


def accuracy_less_than_six(accuracy, line_confusion_matrix):
    line_confusion_matrix.append('medium') if accuracy_grather_than_four_and_less_than_six(
        accuracy) else line_confusion_matrix.append('hard')


def check_accuracy(confusion_matrix, line_confusion_matrix, pos):
    line_confusion_matrix.append('easy') if accuracy_greather_than_six(
        confusion_matrix[pos]) else accuracy_less_than_six(confusion_matrix[pos], line_confusion_matrix)


def size_list_labels_between_six_and_twenty(list_labels):
    return 6 <= len(list_labels) <= 20


def get_size_list_labels_between_six_and_twenty(normalized):
    fontsize_title = 40
    fontsize_tick = 10
    fontsize_label = 20
    figsize = (15, 15)
    if not normalized:
        figsize = (8, 8)
    return figsize, fontsize_title, fontsize_tick, fontsize_label


def size_list_labels_between_twenty_one_and_thirty_four(list_labels):
    return 21 <= len(list_labels) <= 34


def get_size_list_labels_between_twenty_one_and_thirty_four(normalized):
    fontsize_title = 42
    fontsize_tick = 12
    fontsize_label = 24
    figsize = (25, 25)
    if not normalized:
        figsize = (15, 15)
    return figsize, fontsize_title, fontsize_tick, fontsize_label


def size_list_labels_between_thirty_five_and_fifty_five(list_labels):
    return 35 <= len(list_labels) <= 55


def get_size_list_labels_between_thirty_five_and_fifty_five(normalized):
    fontsize_title = 48
    fontsize_tick = 15
    fontsize_label = 30
    figsize = (35, 35)
    if not normalized:
        figsize = (25, 25)
    return figsize, fontsize_title, fontsize_tick, fontsize_label


def size_list_labels_between_one_hundred_and_one_hundred_ten(list_labels):
    return 100 <= len(list_labels) <= 110


def get_size_list_labels_one_hundred_and_one_hundred_ten(normalized):
    fontsize_title = 84
    fontsize_tick = 15
    fontsize_label = 30
    figsize = (75, 75)
    if not normalized:
        figsize = (55, 55)
    return figsize, fontsize_title, fontsize_tick, fontsize_label


def size_list_labels_between_one_hundred_sixty_and_one_hundred_seventy(list_labels):
    return 160 <= len(list_labels) <= 170


def get_size_list_labels_one_hundred_sixty_and_one_hundred_seventy(normalized):
    fontsize_title = 96
    fontsize_tick = 15
    fontsize_label = 30
    figsize = (115, 115)
    if not normalized:
        figsize = (95, 95)
    return figsize, fontsize_title, fontsize_tick, fontsize_label


def size_list_labels_between_two_hundred_thirty_and_two_hundred_fourty(list_labels):
    return 230 <= len(list_labels) <= 240


def get_size_list_labels_two_hundred_thirty_and_two_hundred_fourty(normalized):
    fontsize_title = 160
    fontsize_tick = 15
    fontsize_label = 30
    figsize = (165, 165)
    if not normalized:
        figsize = (145, 145)
    return figsize, fontsize_title, fontsize_tick, fontsize_label


def get_size(list_labels, normalized=True):
    if size_list_labels_between_one_and_five(list_labels):
        return get_size_list_labels_between_one_and_five(normalized)
    elif size_list_labels_between_six_and_twenty(list_labels):
        return get_size_list_labels_between_six_and_twenty(normalized)
    elif size_list_labels_between_twenty_one_and_thirty_four(list_labels):
        return get_size_list_labels_between_twenty_one_and_thirty_four(normalized)
    elif size_list_labels_between_thirty_five_and_fifty_five(list_labels):
        return get_size_list_labels_between_thirty_five_and_fifty_five(normalized)
    elif size_list_labels_between_sixty_and_seventy_one(list_labels):
        return get_size_list_labels_between_sixty_and_seventy_one(normalized)
    elif size_list_labels_between_one_hundred_and_one_hundred_ten(list_labels):
        return get_size_list_labels_one_hundred_and_one_hundred_ten(normalized)
    elif size_list_labels_between_one_hundred_sixty_and_one_hundred_seventy(list_labels):
        return get_size_list_labels_one_hundred_sixty_and_one_hundred_seventy(normalized)
    elif size_list_labels_between_two_hundred_thirty_and_two_hundred_fourty(list_labels):
        return get_size_list_labels_two_hundred_thirty_and_two_hundred_fourty(normalized)


def size_list_labels_between_sixty_and_seventy_one(list_labels):
    return 60 <= len(list_labels) <= 71


def get_size_list_labels_between_sixty_and_seventy_one(normalized):
    fontsize_title = 48
    fontsize_tick = 15
    fontsize_label = 30
    figsize = (45, 45)
    if not normalized:
        figsize = (35, 35)
    return figsize, fontsize_title, fontsize_tick, fontsize_label


def get_size_list_labels_between_one_and_five(normalized):
    fontsize_title = 24
    fontsize_tick = 8
    fontsize_label = 14
    figsize = (5, 5)
    if not normalized:
        figsize = (5, 4)
    return figsize, fontsize_title, fontsize_tick, fontsize_label


def size_list_labels_between_one_and_five(list_labels):
    return 1 <= len(list_labels) <= 5


def save_confusion_matrix_multilabel(list_confusion_matrix, list_labels, path, rule):
    for i, confusion_matrix in enumerate(list_confusion_matrix):
        taxon = list_labels[i]['taxon']
        taxon_italic = list_labels[i]['taxon_italic']
        filename = 'ConfusionMatrix_%s_%s.png' % (rule, taxon)

        path_to_multilabel = os.path.join(path, 'multilabel')
        pathlib.Path(path_to_multilabel).mkdir(exist_ok=True, parents=True)

        filename = os.path.join(path_to_multilabel, filename)
        list_ticklabels = ['False', 'Positive']
        save_confusion_matrix(confusion_matrix, filename, f'Confusion Matrix\n%s' % taxon_italic, fmt='d',
                              xticklabels=list_ticklabels, yticklabels=list_ticklabels, rotation_xtickslabels=0,
                              rotation_ytickslabels=0)
        # save_confusion_matrix_sheet(confusion_matrix, filename.replace('.png', ''), list_ticklabels, list_ticklabels)


def save_confusion_matrix_fold(data, list_labels, path_fold, result, rule):
    path_confusion_matrix = os.path.join(path_fold, 'confusion_matrix', rule)
    pathlib.Path(path_confusion_matrix).mkdir(exist_ok=True, parents=True)

    list_samples_per_label = dict(collections.Counter(result[0]['y_true']))
    yticklabels = get_labels_and_count_samples(list_labels, list_samples_per_label, data['n_patch'])
    xticklabels = get_only_labels(list_labels)

    # ConfusionMatrix
    confusion_matrix = result[0]['confusion_matrix']
    figsize, fontsize_title, fontsize_ticklabels, fontsize_label = get_size(list_labels)
    save_confusion_matrix_normal(confusion_matrix, path_confusion_matrix, rule, xticklabels, yticklabels,
                                 figsize=figsize, fontsize_title=fontsize_title,
                                 fontsize_ticklabels=fontsize_ticklabels, fontsize_label=fontsize_label)

    # ConfusionMatrix normalized
    figsize, fontsize_title, fontsize_ticklabels, fontsize_label = get_size(list_labels, normalized=True)
    confusion_matrix_normalized = result[0]['confusion_matrix_normalized']
    save_confusion_matrix_normalized(confusion_matrix_normalized, path_confusion_matrix, rule, xticklabels, yticklabels,
                                     figsize=figsize, fontsize_title=fontsize_title,
                                     fontsize_ticklabels=fontsize_ticklabels, fontsize_label=fontsize_label)

    # ConfusionMatrix multilabel
    list_confusion_matrix_multilabel = result[0]['confusion_matrix_multilabel']
    save_confusion_matrix_multilabel(list_confusion_matrix_multilabel, list_labels, path_confusion_matrix, rule)

    filename_compress = os.path.join(path_fold, 'confusion_matrix.tar.gz')
    foldername = os.path.join(path_fold, 'confusion_matrix')
    compress_folder(filename_compress, foldername)
    print('[CONFUSION MATRIX] compress folder %s' % foldername)

    shutil.rmtree(foldername)
    print('[CONFUSION MATRIX] delete folder %s' % foldername)


def save_confusion_matrix_normal(confusion_matrix, path_confusion_matrix, rule, xticklabels, yticklabels,
                                 figsize=(5, 5), fontsize_title=24, fontsize_ticklabels=8, fontsize_label=14):
    filename = os.path.join(path_confusion_matrix, 'ConfusionMatrix_%s.png' % rule)
    print(f'[CONFUSION MATRIX] save %s' % filename)
    save_confusion_matrix(confusion_matrix, filename, 'Confusion Matrix', figsize=figsize,
                          fontsize_title=fontsize_title, fontsize_ticklabels=fontsize_ticklabels,
                          fontsize_label=fontsize_label, fmt='.4g', xticklabels=xticklabels, yticklabels=yticklabels,
                          rotation_xtickslabels=90, rotation_ytickslabels=0)


def get_only_labels(list_labels):
    return [label['taxon_italic'] for label in list_labels]


def get_labels_and_count_samples(list_labels, list_samples_per_label, n_patch):
    return [label['taxon_italic'] + ' (' + str(int(list_samples_per_label[label['id']] / n_patch)) + ')'
            for label in list_labels]


def save_confusion_matrix_normalized(confusion_matrix, path, rule, xticklabels, yticklabels, figsize=(5, 5),
                                     fontsize_title=24, fontsize_ticklabels=8, fontsize_label=14):
    filename = os.path.join(path, 'ConfusionMatrix_%s_normalized.png' % rule)
    print(f'[CONFUSION MATRIX] save %s' % filename)
    save_confusion_matrix(confusion_matrix, filename, 'Confusion Matrix', figsize=figsize,
                          fontsize_title=fontsize_title, fontsize_ticklabels=fontsize_ticklabels,
                          fontsize_label=fontsize_label, fmt='.2f',
                          xticklabels=xticklabels, yticklabels=yticklabels, rotation_xtickslabels=90,
                          rotation_ytickslabels=0)


def save_confusion_matrix(confusion_matrix, filename, title, figsize=(5, 5), fontsize_title=24, fontsize_ticklabels=8,
                          fontsize_label=14, fmt='.2g', xticklabels=None, yticklabels=None, rotation_xtickslabels=0,
                          rotation_ytickslabels=90):
    vmin = np.min(confusion_matrix)
    vmax = np.max(confusion_matrix)
    off_diag_mask = np.eye(*confusion_matrix.shape, dtype=bool)

    figure, axis = plt.subplots(figsize=figsize)
    axis = sns.heatmap(confusion_matrix, annot=True, mask=~off_diag_mask, cmap='Reds', fmt=fmt, vmin=vmin, vmax=vmax,
                       ax=axis, annot_kws={'fontweight': 'bold', 'size': 12})
    axis = sns.heatmap(confusion_matrix, annot=True, mask=off_diag_mask, cmap='Reds', fmt=fmt, vmin=vmin, vmax=vmax,
                       cbar=False, ax=axis)

    list_xtick = [i + 0.5 for i in range(len(xticklabels))]
    list_ytick = [i + 0.5 for i in range(len(yticklabels))]

    axis.set_xticks(list_xtick, labels=xticklabels, fontsize=fontsize_ticklabels, rotation=rotation_xtickslabels)
    axis.set_yticks(list_ytick, labels=yticklabels, fontsize=fontsize_ticklabels, rotation=rotation_ytickslabels)
    axis.set_xlabel('True label', fontsize=fontsize_label)
    axis.set_ylabel('Prediction label', fontsize=fontsize_label)
    axis.set_facecolor('white')
    axis.set_title(title, fontsize=fontsize_title, pad=32)

    plt.ioff()
    plt.tight_layout()
    plt.savefig(filename, format='png')
    plt.cla()
    plt.clf()
    plt.close(figure)


def italic_string_plot(string):
    string = string.replace('\"', '')
    return '$\\it{{%s}}$' % string


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
        print('%s not exits exists' % filename)
        raise
