import os
import sys
import time

from classifier import list_classifiers, find_best_classifier_and_params
from data import show_info_data, get_cv, show_info_data_train_test, split_train_test, load_data
from result import insert_result_fold_and_time, get_result
from save.save import create_path_base, save
from save.save_samples import save_info_samples
from save.save_model import save_best_model

from sklearn.svm import SVC

def run_test(cfg, current_datetime, list_labels, list_input, list_extractor, metric, pca, handcraft=False):
    list_data = load_data(cfg, list_extractor, list_input, pca, handcraft=handcraft)
    print('[INFO] tamanho da lista (bytes): %d' % sys.getsizeof(list_data))

    for data in list_data:
        show_info_data(data)
        run_all_classifiers(cfg, current_datetime, data, handcraft, list_labels, metric)

        if len(list_data) > 1:
            list_data = list_data[1:]
            print('[INFO] tamanho da lista: %d' % len(list_data))
            print('[INFO] tamanho da lista (bytes): %d' % sys.getsizeof(list_data))

    del list_data


def run_all_classifiers(cfg, current_datetime, data, handcraft, list_labels, metric):
    for classifier in list_classifiers:
        # best, classifier_name, time_find_best_params = find_best_classifier_and_params(cfg, classifier, data, metric)
        best = {
            'classifier': SVC(random_state=1234, verbose=True, probability=True, cache_size=8000, kernel='linear'),
            'best_params': None
        }
        classifier_name = 'SVC'
        time_find_best_params = -1
        list_result_fold = []
        list_time = []

        path = create_path_base(cfg, classifier_name, current_datetime, data)
        split = get_cv(cfg, data)

        run_folds(best, classifier_name, data, handcraft, list_labels, list_result_fold, list_time, path, split, time_find_best_params)
        save(best['params'], cfg, data, list_labels, list_result_fold, list_time, metric, path)


def run_folds(best, classifier_name, data, handcraft, list_labels, list_result_fold, list_time, path, split, time_find_best_params):
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
        insert_result_fold_and_time(end_time_train_valid, fold, list_result_fold, list_time, result_max_rule, result_prod_rule, result_sum_rule, start_time_train_valid, time_find_best_params)


def check_input(cfg, current_datetime, list_labels, list_input, list_extractor, metric, pca):
    list_dir = [d for d in list_input if os.path.isdir(d) and len(os.listdir(d)) > 0]
    list_files = [file for file in list_input if os.path.isfile(file)]
    if len(list_dir) > 0:
        run_test(cfg, current_datetime, list_labels, list_input, list_extractor, metric, pca)
    if len(list_files) > 0:
        run_test(cfg, current_datetime, list_labels, list_input, list_extractor, metric, pca, handcraft=True)
