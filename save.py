import csv
import os
import pathlib

import pandas as pd

from save_best import save_best
from save_fold import save_fold
from save_mean import save_mean


def create_file_xlsx_and_csv(list_files):
    for file in list_files:
        df = pd.DataFrame(file['values'], file['index'])
        p = os.path.join(file['path'], 'xlsx')
        pathlib.Path(p).mkdir(exist_ok=True, parents=True)
        df.to_excel(os.path.join(p, f'{file["filename"]}.xlsx'), na_rep='', engine='xlsxwriter', header=False)
        df.to_csv(os.path.join(file['path'], f'{file["filename"]}.csv'), sep=';', na_rep='', header=False,
                  quoting=csv.QUOTE_ALL)


def save_info_dataset(data, path):
    index = ['color_mode', 'data_n_features', 'data_n_samples', 'dataset', 'dim_image', 'dir_input', 'extractor',
             'n_patch', 'slice']
    values = [data['color_mode'], data['n_features'], data['n_samples'], data['dataset'], data['image_size'], data['dir'], data['extractor'], data['n_patch'], data['slice_patch']]
    return [{'filename': 'info', 'index': index, 'path': path, 'values': values}]


def save(best_params, cfg, classifier_name, data, list_result_fold, list_time, path):
    list_files_fold = save_fold(cfg, classifier_name, data['dataset'], list_result_fold, list_time, path)
    list_mean_accuracy, list_mean_f1, list_files_mean = save_mean(list_result_fold, list_time, path)
    list_files_best = save_best(best_params, list_mean_accuracy, list_mean_f1, list_result_fold, path)
    list_file_info = save_info_dataset(data, path)
    create_file_xlsx_and_csv(list_files_fold + list_files_mean + list_files_best + list_file_info)


def create_path_base(cfg, classifier_name, current_datetime, data):
    path = os.path.join(cfg['dir_output'], current_datetime, data['dataset'],
                        data['segmented'], data['color_mode'], str(data['image_size']),
                        data['extractor'], classifier_name, 'patch=' + str(data['n_patch']), str(data['n_features']))
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return path
