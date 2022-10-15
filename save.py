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


def save_info_dataset(color_mode, data, dataset, dim, dir_input, extractor, n_patch, path, slice):
    n_features = data['x'].shape[1]
    n_samples = data['x'].shape[0]
    index = ['color_mode', 'data_n_features', 'data_n_samples', 'dataset', 'dim_image', 'dir_input', 'extractor',
             'n_patch', 'slice']
    values = [color_mode, n_features, n_samples, dataset, dim, dir_input, extractor, n_patch, slice]
    return [{'filename': 'info', 'index': index, 'path': path, 'values': values}]


def save(best_params, cfg, classifier_name, color_mode, data, dataset, dim, extractor, file, list_result_fold,
         list_time, n_patch, path, slice):
    list_files_fold = save_fold(cfg, classifier_name, dataset, list_result_fold, list_time, path)
    list_mean_accuracy, list_mean_f1, list_files_mean = save_mean(list_result_fold, list_time, path)
    list_files_best = save_best(best_params, list_mean_accuracy, list_mean_f1, list_result_fold, path)
    list_file_info = save_info_dataset(color_mode, data, dataset, dim, file, extractor, n_patch, path, slice)
    create_file_xlsx_and_csv(list_files_fold + list_files_mean + list_files_best + list_file_info)
