import csv
import os
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics

ROUND_VALUE = 2


def save_mean(best_params, list_result_fold, list_time, path):
    mean_time_train_valid, mean_time_millisec_train_valid, \
        mean_time_min_train_valid, std_time_train_valid = get_mean_std_time('time_train_valid', list_time)
    mean_time_search_best_params, mean_time_millisec_search_best_params, mean_time_min_search_best_params, \
        std_time_search_best_params = get_mean_std_time('time_search_best_params', list_time)

    list_mean_rule = []
    mean_sum, std_sum, mean_f1_score_sum, std_f1_score_sum, mean_std_sum = get_mean_std_by_rule(list_result_fold, 'sum')
    list_mean_rule.append(mean_std_sum)
    mean_max, std_max, mean_f1_score_max, std_f1_score_max, mean_std_max = get_mean_std_by_rule(list_result_fold, 'max')
    list_mean_rule.append(mean_std_max)
    mean_prod, std_prod, mean_f1_score_prod, std_f1_score_prod, mean_std_prod = get_mean_std_by_rule(list_result_fold, 'prod')
    list_mean_rule.append(mean_std_prod)

    best_mean = max(list_mean_rule, key=lambda x: x['mean'])
    best_mean_f1_score = max(list_mean_rule, key=lambda x: x['mean_f1_score'])
    best_fold = max(list_result_fold, key=lambda x: x['accuracy'])
    best_fold_f1_score = max(list_result_fold, key=lambda x: x['f1_score'])
    print(f'best accuracy: {round(float(best_mean["mean"]), 2)}')
    print(f'best f1_score: {round(float(best_mean_f1_score["mean_f1_score"]), 2)}')
    # print(f'best top_k: {mean_std_sum["top_k"]}')
    # print(f'best top_k: {round(float(best_mean_top_k["mean_top_k"]), 2)}')

    index = ['mean_time_train_valid', 'mean_time_millisec_train_valid', 'mean_time_min_train_valid',
             'std_time_train_valid',
             'mean_time_search_best_params', 'mean_time_millisec_search_best_params',
             'mean_time_min_search_best_params', 'std_time_search_best_params',
             'mean_sum', 'std_sum', 'mean_prod', 'std_prod', 'mean_max', 'std_max',
             'mean_f1_score_sum', 'std_f1_score_sum',
             'mean_f1_score_prod', 'std_f1_score_prod',
             'mean_f1_score_max', 'std_f1_score_max',
             'best_mean_rule', 'best_mean', 'best_mean_std',
             'best_f1_score_mean_rule', 'best_f1_score_mean', 'best_f1_score_mean_std',
             'best_fold', 'best_fold_rule', 'best_fold_accuracy',
             'best_fold_f1_score', 'best_fold_f1_score_rule', 'best_fold_f1_score',
             'best_params']
    values = [mean_time_train_valid, mean_time_millisec_train_valid, mean_time_min_train_valid, std_time_train_valid,
              mean_time_search_best_params, mean_time_millisec_search_best_params, mean_time_min_search_best_params,
              std_time_search_best_params,
              mean_sum, std_sum, mean_prod, std_prod, mean_max, std_max,
              mean_f1_score_sum, std_f1_score_sum,
              mean_f1_score_prod, std_f1_score_prod,
              mean_f1_score_max, std_f1_score_max,
              [best_mean['rule']], [best_mean['mean'], round(float(best_mean['mean']), 2)], [best_mean['std'], round(float(best_mean['std']), 2)],
              [best_mean_f1_score['rule']], [best_mean_f1_score['mean_f1_score'], round(float(best_mean_f1_score['mean_f1_score']), 2)],
              [best_mean_f1_score['std_f1_score'], round(float(best_mean_f1_score['std_f1_score']), 2)],
              [best_fold['fold']], [best_fold['rule']], [best_fold['accuracy'], round(best_fold['accuracy'], 2)],
              [best_fold_f1_score['fold']], [best_fold_f1_score['rule']], [best_fold_f1_score['f1_score'], round(best_fold_f1_score['f1_score'], 2)],
              [best_params]]
    df = pd.DataFrame(values, index)
    df.to_excel(os.path.join(path, 'mean.xlsx'), na_rep='', engine='xlsxwriter', header=False)
    df.to_csv(os.path.join(path, 'mean.csv'), sep=';', na_rep='', header=False, quoting=csv.QUOTE_ALL)


def get_mean_std_by_rule(list_result_fold, rule):
    mean = np.mean([r['accuracy'] for r in list(filter(lambda x: x['rule'] == rule, list_result_fold))])
    mean_f1_score = np.mean([r['f1_score'] for r in list(filter(lambda x: x['rule'] == rule, list_result_fold))])
    std = np.std([r['accuracy'] for r in list(filter(lambda x: x['rule'] == rule, list_result_fold))])
    std_f1_score = np.std([r['f1_score'] for r in list(filter(lambda x: x['rule'] == rule, list_result_fold))])
    return [mean, round(float(mean), 2)], [std, round(float(std), 2)],\
           [mean_f1_score, round(float(mean_f1_score), 2)], [std_f1_score, round(float(std_f1_score), 2)], {
        'mean': mean,
        'std': std,
        'rule': rule,
        'mean_f1_score': mean_f1_score,
        'std_f1_score': std_f1_score,
        # 'top_k': top_k
    }


def get_mean_std_time(key, list_time):
    mean_time = np.mean([t[key] for t in list_time])
    round_mean_time = round(float(mean_time), ROUND_VALUE)
    std_time = np.std([t[key] for t in list_time])
    round_std_time = round(float(std_time), ROUND_VALUE)
    mean_time_millisec = mean_time * 1000
    round_mean_time_millisec = round(float(mean_time_millisec), ROUND_VALUE)
    mean_time_min = mean_time / 60
    round_mean_time_min = round(float(mean_time_min), ROUND_VALUE)
    return [mean_time, round_mean_time], \
           [mean_time_millisec, round_mean_time_millisec], \
           [mean_time_min, round_mean_time_min], [std_time, round_std_time]


def create_file_xlsx_and_csv(filename, index, path, values):
    df = pd.DataFrame(values, index)
    p = os.path.join(path, 'xlsx')
    pathlib.Path(p).mkdir(exist_ok=True, parents=True)
    df.to_excel(os.path.join(p, f'{filename}.xlsx'), na_rep='', engine='xlsxwriter', header=False)
    df.to_csv(os.path.join(path, f'{filename}.csv'), sep=';', na_rep='', header=False, quoting=csv.QUOTE_ALL)


def save_fold(cfg, classifier_name, dataset, list_result_fold, list_time, path):
    for fold in range(0, cfg['fold']):
        list_fold = list(filter(lambda x: x['fold'] == fold, list_result_fold))
        time_fold = list(filter(lambda x: x['fold'] == fold, list_time))

        path_fold = os.path.join(path, str(fold))
        pathlib.Path(path_fold).mkdir(parents=True, exist_ok=True)

        index, values_mean = find_values_fold_by_rule(classifier_name, dataset, list_fold, 'accuracy', path_fold)
        index, values_f1 = find_values_fold_by_rule(classifier_name, dataset, list_fold, 'f1_score', path_fold)

        best_rule_accuracy = max(list_fold, key=lambda x: x['accuracy'])
        create_file_xlsx_and_csv('accuracy', index, path_fold, values_mean)
        create_file_xlsx_and_csv('f1_score', index, path_fold, values_mean)
        create_file_info_fold(best_rule_accuracy, path_fold, time_fold)


def find_values_fold_by_rule(classifier_name, dataset, list_fold, metric, path_fold):
    index = []
    values = []
    for rule in ['max', 'prod', 'sum']:
        result = list(filter(lambda x: x['rule'] == rule, list_fold))
        index.append(rule)
        values.append([result[0][metric], round(result[0][metric], ROUND_VALUE)])
        save_confusion_matrix(classifier_name, dataset, path_fold, result[0])
    return index, values


def create_file_info_fold(best_rule, path, time):
    index_time, values_time = info_time(time)
    index_best, values_best = info_best(best_rule)
    index_best_f1, values_best_f1 = info_best_f1(best_rule)

    create_file_xlsx_and_csv('fold_time', index_time, path, values_time)
    create_file_xlsx_and_csv('fold_best', index_best, path, values_best)
    create_file_xlsx_and_csv('fold_best_f1', index_best_f1, path, values_best_f1)


def info_best_f1(best):
    best_rule = [best['rule']]
    best_f1 = best['f1_score']
    round_best_f1 = round(float(best_f1), ROUND_VALUE)
    index = ['best_rule', 'best_f1']
    values = [best_rule, [best_f1, round_best_f1]]
    return index, values


def info_best(best):
    best_rule = [best['rule']]
    best_accuracy = best['accuracy']
    round_best_accuracy = round(float(best_accuracy), ROUND_VALUE)
    index_best = ['best_rule', 'best_accuracy']
    values_best = [best_rule, [best_accuracy, round_best_accuracy]]
    return index_best, values_best


def info_time(time):
    time_train_valid = time[0]['time_train_valid']
    round_time_train_valid = round(time_train_valid, ROUND_VALUE)
    time_search_best_params = time[0]['time_search_best_params']
    round_search_best_params = round(time_search_best_params, ROUND_VALUE)
    index_time = ['time_train_valid', 'time_search_best_params']
    values_time = [[time_train_valid, round_time_train_valid],
                   [time_search_best_params, round_search_best_params]]
    return index_time, values_time


def save_info_dataset(color_mode, data, dataset, dim, dir_input, extractor, n_patch, path, slice):
    n_features = data['x'].shape[1]
    n_samples = data['x'].shape[0]
    index = ['color_mode', 'data_n_features', 'data_n_samples', 'dataset', 'dim_image', 'dir_input', 'extractor', 'n_patch',
             'slice']
    values = [color_mode, n_features, n_samples, dataset, dim, dir_input, extractor, n_patch, slice]
    create_file_xlsx_and_csv('info', index, path, values)


def save_confusion_matrix(classifier_name, dataset, path, result):
    filename = f'confusion_matrix_{result["rule"]}.png'
    # cinco labels -> IWSSIP
    # labels = ['$\it{Manekia}$', '$\it{Ottonia}$', '$\it{Peperomia}$', '$\it{Piper}$', '$\it{Pothomorphe}$']

    # acima de cinco labels -> dataset George
    # labels =

    # acima de cinco dez -> dataset George

    # acima de cinco vinte -> dataset George


    # duas labels -> dataset George
    labels = ['$\it{Peperomia}$', '$\it{Piper}$']

    confusion_matrix = sklearn.metrics.ConfusionMatrixDisplay(result['confusion_matrix'])
    confusion_matrix.plot(cmap='Reds')

    title = f'Confusion Matrix\ndataset: {dataset}, classifier: {classifier_name}\naccuracy: {round(result["accuracy"], ROUND_VALUE)}, rule: {result["rule"]}'
    fontsize_title = 12
    pad_title = 20

    fontsize_labels = 8

    background_color = 'white'
    plt.ioff()
    plt.title(title, fontsize=fontsize_title, pad=pad_title)
    plt.xticks(np.arange(len(labels)), labels, rotation=45, fontsize=fontsize_labels)
    plt.yticks(np.arange(len(labels)), labels, fontsize=fontsize_labels)
    plt.ylabel('y_test', fontsize=fontsize_labels)
    plt.xlabel('y_pred', fontsize=fontsize_labels)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.25)
    plt.rcParams['figure.facecolor'] = background_color
    plt.tight_layout()
    plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()


def save(best_params, cfg, classifier_name, color_mode, data, dataset, dim, extractor, file, list_result_fold,
         list_time, n_patch, path, slice):
    save_fold(cfg, classifier_name, dataset, list_result_fold, list_time, path)
    save_mean(best_params, list_result_fold, list_time, path)
    save_info_dataset(color_mode, data, dataset, dim, file, extractor, n_patch, path, slice)
