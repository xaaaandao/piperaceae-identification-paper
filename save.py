import csv
import os
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics


def save_mean(best_params, list_result_fold, list_time, path):
    mean_time_train_valid, mean_time_millisec_train_valid, \
        mean_time_min_train_valid, std_time_train_valid = get_mean_std_time('time_train_valid', list_time)
    mean_time_search_best_params, mean_time_millisec_search_best_params, mean_time_min_search_best_params, \
        std_time_search_best_params = get_mean_std_time('time_search_best_params', list_time)

    list_mean_rule = []
    mean_sum, std_sum, mean_std_sum = get_mean_std_by_rule(list_result_fold, 'sum')
    list_mean_rule.append(mean_std_sum)
    mean_max, std_max, mean_std_max = get_mean_std_by_rule(list_result_fold, 'max')
    list_mean_rule.append(mean_std_max)
    mean_prod, std_prod, mean_std_prod = get_mean_std_by_rule(list_result_fold, 'prod')
    list_mean_rule.append(mean_std_prod)

    best_mean = max(list_mean_rule, key=lambda x: x['mean'])
    best_fold = max(list_result_fold, key=lambda x: x['accuracy'])
    print(f'best accuracy: {round(float(best_mean["mean"]), 2)}')

    index = ['mean_time_train_valid', 'mean_time_millisec_train_valid', 'mean_time_min_train_valid',
             'std_time_train_valid',
             'mean_time_search_best_params', 'mean_time_millisec_search_best_params',
             'mean_time_min_search_best_params', 'std_time_search_best_params',
             'mean_sum', 'std_sum', 'mean_prod', 'std_prod', 'mean_max', 'std_max',
             'best_mean_rule', 'best_mean', 'best_mean_std',
             'best_fold', 'best_fold_rule', 'best_fold_accuracy',
             'best_params']
    values = [mean_time_train_valid, mean_time_millisec_train_valid, mean_time_min_train_valid, [std_time_train_valid],
              mean_time_search_best_params, mean_time_millisec_search_best_params, mean_time_min_search_best_params,
              [std_time_search_best_params],
              mean_sum, std_sum, mean_prod, std_prod, mean_max, std_max,
              [best_mean['rule']], [best_mean['mean'], round(float(best_mean['mean']), 2)],
              [best_mean['std'], round(float(best_mean['std']), 2)],
              [best_fold['fold']], [best_fold['rule']], [best_fold['accuracy'], round(best_fold['accuracy'], 2)],
              [best_params]]
    dataframe_mean = pd.DataFrame(values, index)
    dataframe_mean.to_csv(os.path.join(path, 'mean.csv'), sep=';', na_rep='', header=False, quoting=csv.QUOTE_ALL)


def get_mean_std_by_rule(list_result_fold, rule):
    mean = np.mean([r['accuracy'] for r in list(filter(lambda x: x['rule'] == rule, list_result_fold))])
    std = np.std([r['accuracy'] for r in list(filter(lambda x: x['rule'] == rule, list_result_fold))])
    return [mean, round(float(mean), 2)], [std, round(float(std), 2)], {
        'mean': mean,
        'std': std,
        'rule': rule
    }


def get_mean_std_time(key, list_time):
    mean_time = np.mean([t[key] for t in list_time])
    std_time = np.std([t[key] for t in list_time])
    mean_time_millisec = mean_time * 1000
    mean_time_min = mean_time / 60
    return [mean_time, round(float(mean_time), 2)], \
           [mean_time_millisec, round(float(mean_time_millisec), 2)], \
           [mean_time_min, round(float(mean_time_min), 2)], [std_time, round(float(std_time), 2)]


def save_fold(cfg, classifier_name, dataset, list_result_fold, list_time, path):
    for fold in range(0, cfg['fold']):
        list_fold = list(filter(lambda x: x['fold'] == fold, list_result_fold))
        time_fold = list(filter(lambda x: x['fold'] == fold, list_time))

        path_fold = os.path.join(path, str(fold))
        pathlib.Path(path_fold).mkdir(parents=True, exist_ok=True)

        index = []
        values = []
        for rule in ['max', 'prod', 'sum']:
            result = list(filter(lambda x: x['rule'] == rule, list_fold))
            index.append(rule)
            values.append([result[0]['accuracy'], round(result[0]['accuracy'], 2)])
            save_confusion_matrix(classifier_name, dataset, path_fold, result[0])

        best_rule = max(list_fold, key=lambda x: x['accuracy'])
        create_file_accuracy_by_rule(index, path_fold, values)
        create_file_info_fold(best_rule, path_fold, time_fold)


def create_file_info_fold(best_rule, path, time):
    index_time = ['time_train_valid', 'time_search_best_params']
    values_time = [[time[0]['time_train_valid'], round(time[0]['time_train_valid'], 2)],
                   [time[0]['time_search_best_params'], round(time[0]['time_search_best_params'], 2)]]

    dataframe_time = pd.DataFrame(values_time, index_time)

    index_best = ['best_rule', 'best_accuracy']
    values_best = [[best_rule['rule']], [best_rule['accuracy'], round(float(best_rule['accuracy']), 2)]]
    dataframe_best_rule = pd.DataFrame(values_best, index_best)

    dataframe_info = pd.concat([dataframe_time, dataframe_best_rule])
    dataframe_info.to_csv(os.path.join(path, 'fold_info.csv'), sep=';', na_rep='', header=False,
                          quoting=csv.QUOTE_ALL)


def create_file_accuracy_by_rule(index, path, values):
    dataframe_fold = pd.DataFrame(values, index)
    dataframe_fold.to_csv(os.path.join(path, 'accuracy_by_rule.csv'), sep=';', na_rep='', header=False,
                          quoting=csv.QUOTE_ALL)


def save_info_dataset(color_mode, data, dataset, dim, dir_input, extractor, n_patch, path, slice):
    index = ['color_mode', 'data_n_features', 'data_n_samples', 'dataset', 'dim_image', 'dir_input', 'extractor', 'n_patch',
             'slice']
    values = [color_mode, data['x'].shape[1], data['x'].shape[0], dataset, dim, dir_input, extractor, n_patch, slice]
    dataframe = pd.DataFrame(values, index)
    dataframe.to_csv(os.path.join(path, 'info.csv'), sep=';', na_rep='', header=False, quoting=csv.QUOTE_ALL)


def save_confusion_matrix(classifier_name, dataset, path, result):
    filename = f'confusion_matrix-{result["rule"]}.png'
    labels = ['$\it{Manekia}$', '$\it{Ottonia}$', '$\it{Peperomia}$', '$\it{Piper}$', '$\it{Pothomorphe}$']
    confusion_matrix = sklearn.metrics.ConfusionMatrixDisplay(result['confusion_matrix'])
    confusion_matrix.plot(cmap='Reds')
    title = f'Confusion Matrix\ndataset: {dataset}, classifier: {classifier_name}\naccuracy: {round(result["accuracy"], 2)}, rule: {result["rule"]}'
    plt.ioff()
    plt.title(title, fontsize=12, pad=20)
    plt.xticks(np.arange(5), labels, rotation=45, fontsize=8)
    plt.yticks(np.arange(5), labels, fontsize=8)
    plt.ylabel('y_test', fontsize=8)
    plt.xlabel('y_pred', fontsize=8)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.25)
    plt.rcParams['figure.facecolor'] = 'white'
    # plt.rcParams['figure.figsize'] = (10, 10)
    plt.tight_layout()
    plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    plt.cla()
    plt.clf()
    # time.sleep(3) # avoid bug first plot
    plt.close()
