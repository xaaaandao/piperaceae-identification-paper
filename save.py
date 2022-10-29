import collections
import csv
import os
import pandas as pd
import pathlib

from save_best import save_best
from save_fold import save_fold
from save_mean import save_mean


def create_file_xlsx_and_csv(list_files):
    for file in list_files:
        df = pd.DataFrame(file['values'], file['index'])
        p = os.path.join(file['path'], 'xlsx')
        pathlib.Path(p).mkdir(exist_ok=True, parents=True)
        filename = file['filename']
        df.to_excel(os.path.join(p, f'{filename}.xlsx'), na_rep='', engine='xlsxwriter', header=False)
        df.to_csv(os.path.join(file['path'], f'{filename}.csv'), sep=';', na_rep='', header=False,
                  quoting=csv.QUOTE_ALL)
        df.to_excel(os.path.join(p, f'{file["filename"]}.xlsx'), na_rep='', engine='xlsxwriter', header=False)
        df.to_csv(os.path.join(file['path'], f'{file["filename"]}.csv'), sep=';', na_rep='', header=False,
                  quoting=csv.QUOTE_ALL)


def save_info_dataset(data, metric, path):
    index = ['color_mode', 'data_n_features', 'data_n_samples', 'dataset', 'dim_image', 'dir_input', 'extractor',
             'n_patch', 'slice', 'metric']
    values = [data['color_mode'], data['n_features'], data['n_samples'], data['dataset'], data['image_size'],
              data['dir'], data['extractor'], data['n_patch'], data['slice_patch'], metric]
    return [{'filename': 'info', 'index': index, 'path': path, 'values': values}]


def save(best_params, cfg, classifier_name, data, labels, list_result_fold, list_time, metric, path):
    list_files_fold = save_fold(cfg, classifier_name, data['dataset'], labels, list_result_fold, list_time, path)
    list_mean_accuracy, list_mean_f1, list_files_mean = save_mean(list_result_fold, list_time, path)
    list_files_best = save_best(best_params, list_mean_accuracy, list_mean_f1, list_result_fold, path)
    list_file_info = save_info_dataset(data, metric, path)
    create_file_xlsx_and_csv(list_files_fold + list_files_mean + list_files_best + list_file_info)


def create_path_base(cfg, classifier_name, current_datetime, data):
    path = os.path.join(cfg['dir_output'], current_datetime, data['dataset'],
                        data['segmented'], data['color_mode'], str(data['image_size']),
                        data['extractor'], classifier_name, 'patch=' + str(data['n_patch']), str(data['n_features']))
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return path


def save_info_samples(fold, labels, index_train, index_test, n_patch, path, y, y_train, y_test):
    p = os.path.join(path, str(fold), 'info_samples')
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

    l = []
    for k, v in collections.Counter(y_train).items():
        result_y = {'value': v2 for k2, v2 in collections.Counter(y).items() if k2 == k }
        result_y_test = {'value': v2 for k2, v2 in collections.Counter(y_test).items() if k2 == k }
        percentage_train = round((v * 100) / result_y['value'], 2)
        percentage_test = round((result_y_test['value'] * 100) / result_y['value'], 2)
        samples_train = int(v/n_patch)
        samples_test = int(result_y_test['value']/n_patch)
        samples_total = int(result_y['value']/n_patch)
        clabels = labels[int(k)-1].replace('$\it{', '').replace('}$', '')
        l.append({'label': f'{k} ({clabels})', 'samples_train': f'{v} ({samples_train}) ({percentage_train})', 'samples_test': f'{result_y_test["value"]} ({samples_test})({percentage_test})', 'total': f'{result_y["value"]} ({samples_total})'})

    df = pd.DataFrame(l)
    df.to_csv(os.path.join(p, 'samples_fold.csv'), sep=';', na_rep='', index=False, quoting=csv.QUOTE_ALL)
    df.to_excel(os.path.join(p, 'samples_fold.xlsx'), na_rep='', engine='xlsxwriter', index=False)

    samples_used = {'index_train': [index_train], 'index_test': [index_test]}
    df = pd.DataFrame(samples_used)
    df = df.transpose()
    df.to_csv(os.path.join(path, str(fold), 'samples_used.csv'), sep=';', na_rep='', header=False, quoting=csv.QUOTE_ALL)
    df.to_excel(os.path.join(path, str(fold), 'samples_used.xlsx'), na_rep='', engine='xlsxwriter', header=False)
