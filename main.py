import os
import time

import click
import datetime

from classifier import list_classifiers, find_best_classifier_and_params
from data import show_info_data, get_cv, show_info_data_train_test
from handcraft import handcraft
from non_handcraft import non_handcraft, load_data, split_train_test, get_result
from confusion_matrix import get_list_label
from result import insert_result_fold_and_time
from save import create_path_base, save_info_samples, save
from save_model import save_best_model

cfg = {
    'fold': 5,
    'n_jobs': -1,
    'seed': 1234,
    'dir_input': '../dataset/features',
    'dir_output': './out',
    'verbose': 42
}

list_extractor = {
    'lbp': [59],
    'surf64': [128, 256, 257],
    'surf128': [128, 256, 513],
    'mobilenetv2': [128, 256, 512, 1024, 1280],
    'resnet50v2': [128, 256, 512, 1024, 2048],
    'vgg16': [128, 256, 512]
}

@click.command()
@click.option('-i', '--list_user_input', multiple=True, default=['/home/xandao/Documentos/resultados_gimp/identificacao_george/especie/20'])
@click.option('-l', '--labels', default=['/home/xandao/Documentos/GitHub/dataset_gimp/imagens_george/imagens/RGB/specific_epithet/256/20/label2.txt'])
@click.option('-m', '--metric', type=click.Choice(['f1_weighted', 'accuracy']), default='f1_weighted')
def main(list_user_input, labels, metric):
    current_datetime = datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
    list_data_input = []
    if len(list_user_input) == 0 and len(list_data_input) == 0:
        raise ValueError(f'list data input is empty')

    list_data_input = list_data_input + [i for i in list(list_user_input) if i not in list_data_input]
    print(f'[INFO] quantidade de entradas: {len(list_data_input)}, filename labels: {labels}')
    list_labels = get_list_label(labels)

    if len(list_data_input) > 0 and len(list_labels) > 0:
        check_input(cfg, current_datetime, list_labels, list_data_input, list_extractor, metric)
        # handcraft(cfg, current_datetime, list_labels, list_data_input, list_extractor, metric)
        # non_handcraft(cfg, current_datetime, list_labels, list_data_input, list_extractor, metric)


def run_test(cfg, current_datetime, list_labels, list_input, list_extractor, metric, handcraft=False):
    list_data = load_data(cfg, list_extractor, list_input, handcraft=handcraft)

    for data in list_data:
        show_info_data(data)

        for classifier in list_classifiers:
            best, classifier_name, time_find_best_params = find_best_classifier_and_params(cfg, classifier, data, metric)
            list_result_fold = []
            list_time = []

            path = create_path_base(cfg, classifier_name, current_datetime, data)
            split = get_cv(cfg, data)

            for fold, (index_train, index_test) in enumerate(split):
                x_test, x_train, y_test, y_train = split_train_test(data, index_test, index_train, handcraft=handcraft)

                show_info_data_train_test(classifier_name, fold, x_test, x_train, y_test, y_train)

                start_time_train_valid = time.time()
                best['classifier'].fit(x_train, y_train)
                y_pred = best['classifier'].predict_proba(x_test)

                save_info_samples(fold, list_labels, index_train, index_test, data['n_patch'], path, data['y'], y_train, y_test)
                save_best_model(best['classifier'], fold, path)

                result_max_rule, result_prod_rule, result_sum_rule = get_result(data, fold, list_labels, y_pred, y_test, handcraft=handcraft)

                end_time_train_valid = time.time()
                insert_result_fold_and_time(end_time_train_valid, fold, list_result_fold, list_time, result_max_rule,
                                            result_prod_rule, result_sum_rule, start_time_train_valid, time_find_best_params)
            save(best['params'], cfg, classifier_name, data, list_labels, list_result_fold, list_time, metric, path)


def check_input(cfg, current_datetime, list_labels, list_input, list_extractor, metric):
    list_dir = [d for d in list_input if os.path.isdir(d) and len(os.listdir(d)) > 0]
    list_files = [file for file in list_input if os.path.isfile(file)]
    if len(list_dir) > 0:
        run_test(cfg, current_datetime, list_labels, list_input, list_extractor, metric)
    if len(list_files) > 0:
        run_test(cfg, current_datetime, list_labels, list_input, list_extractor, metric, handcraft=True)


if __name__ == '__main__':
    main()
