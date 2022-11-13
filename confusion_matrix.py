import collections
import csv
import os
import pathlib

import pandas as pd

from save_fold import get_labels_and_count_samples, get_only_labels, save_confusion_matrix_normalized, \
    save_confusion_matrix


def save_confusion_matrix_sheet(confusion_matrix, filename, xticklabels, yticklabels):
    index = [label.replace('$\it{', '').replace('}$', '') for label in yticklabels]
    columns = [label.replace('$\it{', '').replace('}$', '') for label in xticklabels]
    # confusion_matrix = confusion_matrix.astype('object')
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


def confusion_matrix_by_fold(classifier_name, data, list_labels, list_fold, path_fold):
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
            save_confusion_matrix_sheet(result[0]['confusion_matrix'], os.path.join(path_confusion_matrix, 'confusionmatrix_normalized_'), xticklabels, yticklabels)


def save_confusion_matrix_normal(confusion_matrix, path_confusion_matrix, rule, xticklabels, yticklabels):
    filename = os.path.join(path_confusion_matrix, f'ConfusionMatrix_{rule}.png')
    print(f'save {filename}')
    save_confusion_matrix(confusion_matrix, filename, 'Confusion Matrix', figsize=(15, 15), fmt='.2g',
                          xticklabels=xticklabels, yticklabels=yticklabels, rotation_xtickslabels=90,
                          rotation_ytickslabels=0)
