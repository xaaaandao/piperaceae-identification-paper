import os
import pathlib

from save.save_best import save_info_best
from save.save_dataset import save_info_dataset
from save.save_fold import save_fold
from save.save_mean import save_mean


def save(best_params, cfg, data, labels, list_result_fold, list_time, metric, path):
    save_fold(cfg, data, labels, list_result_fold, list_time, path)
    list_mean_accuracy, list_mean_f1 = save_mean(list_result_fold, list_time, path)
    save_info_best(best_params, list_mean_accuracy, list_mean_f1, list_result_fold, path)
    save_info_dataset(data, metric, path)


def create_path_base(cfg, classifier_name, current_datetime, data):
    path = os.path.join(cfg['dir_output'], current_datetime, data['dataset'],
                        data['segmented'], data['color_mode'], str(data['image_size']),
                        data['extractor'], classifier_name, 'patch=%s' % str(data['n_patch']), str(data['n_features']))
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return path


