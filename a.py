import collections
import csv
import os
import pathlib
import tarfile
import time

import joblib
import numpy as np
import pandas as pd

from data import get_info, merge_all_files_of_dir, get_x_y, get_cv, get_samples_with_patch, show_info_data_train_test, \
    show_info_data
from main import cfg, list_extractor
from result import calculate_test, insert_result_fold_and_time
from save import save_info_samples, save
from save_fold import get_list_label, save_confusion_matrix
from save_model import save_best_model


def get_model(path):
    file = tarfile.open(path)
    file.extractall()
    filename_pkl = file.getnames()[0]
    file.close()
    clf = joblib.load(filename_pkl)

    best = {
        'classifier': clf,
        'params': clf.get_params()
    }
    print(clf.get_params())
    classifier_name = clf.__class__.__name__
    time_find_best_params = 0
    return filename_pkl, best, classifier_name, time_find_best_params


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


def save_others(list_labels, n_patch, path, result, y_test):
    list_confusion_matrix = result['confusion_matrix_multilabel']
    p = os.path.join(path, result['rule'], 'new')
    pathlib.Path(p).mkdir(exist_ok=True, parents=True)
    rule = result['rule']

    list_labels = sorted(list_labels, key=lambda d: d['id'])
    for i, confusion_matrix in enumerate(list_confusion_matrix):
        taxon = list_labels[i]['taxon']
        taxon_italic = list_labels[i]['taxon_italic']
        filename = 'confusion_matrix_' + taxon + '_' + rule + '.png'
        path_to_multilabel = os.path.join(p, 'multilabel')
        pathlib.Path(path_to_multilabel).mkdir(exist_ok=True, parents=True)
        path_to_csv_xlsx = os.path.join(path_to_multilabel, 'csv_xlsx')
        pathlib.Path(path_to_csv_xlsx).mkdir(exist_ok=True, parents=True)
        filename = os.path.join(path_to_multilabel, filename)
        ticklabels = ['False', 'Positive']
        print(f'save {filename}')
        save_confusion_matrix(confusion_matrix, filename, f'Confusion Matrix\n{taxon_italic}', fmt='d', xticklabels=ticklabels, yticklabels=ticklabels, rotation_xtickslabels=0, rotation_ytickslabels=0)
        save_confusion_matrix_sheet(confusion_matrix, filename.replace('.png', ''), ticklabels, ticklabels)

    list_samples_per_label = dict(collections.Counter(y_test))
    yticklabels = [label['taxon_italic'] + ' (' + str(int(list_samples_per_label[label['id']] / n_patch)) + ')' for label in list_labels]
    xticklabels = [label['taxon_italic'] for label in list_labels]

    confusion_matrix = result['confusion_matrix']
    filename = 'confusion_matrix_' + rule + '.png'
    filename = os.path.join(p, filename)
    print(f'save {filename}')
    save_confusion_matrix(confusion_matrix, filename, 'Confusion Matrix', figsize=(12, 12), fmt='.2g',
                          xticklabels=xticklabels, yticklabels=yticklabels, rotation_xtickslabels=90,
                          rotation_ytickslabels=0)
    # save_confusion_matrix_sheet(confusion_matrix, filename.replace('.png', ''), xticklabels, yticklabels)

    confusion_matrix = result['confusion_matrix_normalized']
    filename = os.path.join(p, f'cf_normalized_{rule}.png')
    print(f'save {filename}')
    save_confusion_matrix(confusion_matrix, filename, 'Confusion Matrix', figsize=(12, 12), fmt='.2f',
                          xticklabels=xticklabels, yticklabels=yticklabels, rotation_xtickslabels=90,
                          rotation_ytickslabels=0)
    save_confusion_matrix_sheet(confusion_matrix, filename.replace('.png', ''), xticklabels, yticklabels)


def main():
    p = '/home/xandao/Documentos/resultados_gimp/identificacao_george/especie/20'
    print(os.path.exists(p))

    if not os.path.exists(p):
        raise IsADirectoryError(f'dir is not found {p}')

    list_info_file = [c for c in pathlib.Path(p).rglob('info.csv') if c.is_file()]

    print(len(list_info_file))
    for l in list_info_file:
        file_info = pd.read_csv(l, sep=';', index_col=0, header=None)
        color_mode = file_info.loc['color_mode', 1]
        dir_data = file_info.loc['dir_input', 1]
        metric = file_info.loc['metric', 1]
        n_features = file_info.loc['data_n_features', 1]
        dataset = None
        image_size = None

        list_data = []
        list_only_dir = [d for d in [dir_data] if os.path.isdir(d) and len(os.listdir(d)) > 0]
        for d in list_only_dir:
            dataset, color_mode, segmented, image_size, extractor, slice_patch = get_info(d)
            data, n_patch = merge_all_files_of_dir(d)
            get_x_y(cfg, color_mode, np.array(data), dataset, extractor, d, image_size, list_data, list_extractor,
                    n_patch, segmented, slice_patch)

        filename_label = f'/home/xandao/Documentos/GitHub/dataset_gimp/imagens_george/imagens/{color_mode.upper()}/specific_epithet/{image_size}/20/label2.txt'
        labels = get_list_label(filename_label)

        list_data = [d for d in list_data if d['n_features'] == int(n_features)]
        for data in list_data:
            show_info_data(data)
            list_result_fold = []
            list_time = []
            split = get_cv(cfg, data)

            for fold, (index_train, index_test) in enumerate(split):
                path = str(l).replace('info.csv', '')
                path_fold = os.path.join(path, str(fold))
                p = os.path.join(path_fold, 'best_model.tar.gz')

                if not os.path.exists(p):
                    raise FileNotFoundError(f'best_model.tar.gz not found')

                path_model, best, classifier_name, time_find_best_params = get_model(p)
                x_train, y_train = get_samples_with_patch(data['x'], data['y'], index_train, data['n_patch'])
                x_test, y_test = get_samples_with_patch(data['x'], data['y'], index_test, data['n_patch'])

                show_info_data_train_test(classifier_name, fold, x_test, x_train, y_test, y_train)

                start_time_train_valid = time.time()
                best['classifier'].fit(x_train, y_train)
                y_pred = best['classifier'].predict_proba(x_test)

                result_max_rule, result_prod_rule, result_sum_rule = calculate_test(fold, labels, y_pred, y_test,
                                                                                    n_patch=int(data['n_patch']))

                end_time_train_valid = time.time()
                insert_result_fold_and_time(end_time_train_valid, fold, list_result_fold, list_time, result_max_rule,
                                            result_prod_rule, result_sum_rule, start_time_train_valid,
                                            time_find_best_params)

                save_others(labels, int(data['n_patch']), path_model.replace('best_model.pkl', ''), result_sum_rule,
                            y_test)
        # break


if __name__ == '__main__':
    main()