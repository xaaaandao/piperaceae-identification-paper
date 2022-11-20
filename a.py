import collections
import os
import pathlib
import tarfile
import time

import click
import joblib
import pandas as pd

from save.save_confusion_matrix import save_confusion_matrix_sheet, get_only_labels, \
    get_labels_and_count_samples, save_confusion_matrix, get_list_label
from data import get_cv, show_info_data_train_test, \
    show_info_data, split_train_test
from main import cfg, list_extractor
from handcraft import load_all_files_npy
from result import calculate_test, insert_result_fold_and_time


def get_model(path):
    filename = get_path_to_model_extracted(path)
    clf = joblib.load(filename)

    best = {
        'classifier': clf,
        'params': clf.get_params()
    }
    classifier_name = clf.__class__.__name__
    time_find_best_params = 0
    return filename, best, classifier_name, time_find_best_params


def get_path_to_model_extracted(path):
    f = tarfile.open(path)
    f.extractall()
    filename = f.getnames()[0]
    f.close()
    return filename


def save_confusion_matrix_multilabel(list_confusion_matrix, list_labels, path, rule):
    for i, confusion_matrix in enumerate(list_confusion_matrix):
        taxon = list_labels[i]['taxon']
        taxon_italic = list_labels[i]['taxon_italic']
        filename = 'confusion_matrix_' + taxon + '_' + rule + '.png'

        path_to_multilabel = os.path.join(path, 'multilabel')
        pathlib.Path(path_to_multilabel).mkdir(exist_ok=True, parents=True)

        path_to_csv_xlsx = os.path.join(path_to_multilabel, 'csv_xlsx')
        pathlib.Path(path_to_csv_xlsx).mkdir(exist_ok=True, parents=True)

        filename = os.path.join(path_to_multilabel, filename)
        list_ticklabels = ['False', 'Positive']
        print(f'[CONFUSION MATRIX] save {filename}')
        save_confusion_matrix(confusion_matrix, filename, f'Confusion Matrix\n{taxon_italic}', fmt='d',
                              xticklabels=list_ticklabels, yticklabels=list_ticklabels, rotation_xtickslabels=0,
                              rotation_ytickslabels=0)
        save_confusion_matrix_sheet(confusion_matrix, filename.replace('.png', ''), list_ticklabels, list_ticklabels)


def save_others(list_labels, n_patch, path, result, y_test):
    list_confusion_matrix = result['confusion_matrix_multilabel']
    p = os.path.join(path, result['rule'], 'new')
    pathlib.Path(p).mkdir(exist_ok=True, parents=True)
    rule = result['rule']

    list_labels = sorted(list_labels, key=lambda d: d['id'])
    save_confusion_matrix_multilabel(list_confusion_matrix, list_labels, p, rule)

    list_samples_per_label = dict(collections.Counter(y_test))
    yticklabels = get_labels_and_count_samples(list_labels, list_samples_per_label, n_patch)
    xticklabels = get_only_labels(list_labels)

    confusion_matrix = result['confusion_matrix']
    # save_confusion_matrix_normal(confusion_matrix, p, rule, xticklabels, yticklabels)
    # save_confusion_matrix_normalized(result['confusion_matrix_normalized'], p, rule, xticklabels,
    #                                  yticklabels)
    confusion_matrix_normalized = result['confusion_matrix_normalized']
    filename = os.path.join(p, 'ConfusionMatrix_Normalized_' + rule)
    save_confusion_matrix_sheet(confusion_matrix_normalized, filename, xticklabels, yticklabels)


@click.command()
@click.option('-path', '--path', required=True, default='/home/xandao/Documentos/resultados_gimp/identificacao_george/especie/20')
@click.option('-l', '--labels', required=True, default='/home/xandao/Documentos/GitHub/dataset_gimp/imagens_george/imagens/RGB/specific_epithet/256/20/label2.txt')
def main2(labels, path):
    if not os.path.exists(path):
        raise IsADirectoryError(f'dir is not found {path}')

    # if not os.path.isfile(path):
    #     raise FileNotFoundError(f'file is not found {path}')

    list_info_file = [c for c in pathlib.Path(path).rglob('info.csv') if c.is_file()]

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
        load_all_files_npy(cfg, list_data, list_extractor, list_only_dir)

        list_labels = get_list_label(labels)

        list_data = [d for d in list_data if d['n_features'] == int(n_features)]
        for data in list_data:
            show_info_data(data)
            list_result_fold = []
            list_time = []
            split = get_cv(cfg, data)

            for fold, (index_train, index_test) in enumerate(split):
                path = str(l).replace('info.csv', '')
                path_fold = os.path.join(path, str(fold))
                path_model = os.path.join(path_fold, 'best_model.tar.gz')

                if not os.path.exists(path_model):
                    raise FileNotFoundError(f'best_model.tar.gz not found')

                path_model, best, classifier_name, time_find_best_params = get_model(path_model)
                x_test, x_train, y_test, y_train = split_train_test(data, index_test, index_train)

                show_info_data_train_test(classifier_name, fold, x_test, x_train, y_test, y_train)

                start_time_train_valid = time.time()
                best['classifier'].fit(x_train, y_train)
                y_pred = best['classifier'].predict_proba(x_test)

                result_max_rule, result_prod_rule, result_sum_rule = calculate_test(fold, list_labels, y_pred, y_test,
                                                                                    n_patch=int(data['n_patch']))

                end_time_train_valid = time.time()
                insert_result_fold_and_time(end_time_train_valid, fold, list_result_fold, list_time, result_max_rule,
                                            result_prod_rule, result_sum_rule, start_time_train_valid,
                                            time_find_best_params)

                save_others(list_labels, int(data['n_patch']), path_model.replace('best_model.pkl', ''), result_sum_rule,
                            y_test)


if __name__ == '__main__':
    main2()