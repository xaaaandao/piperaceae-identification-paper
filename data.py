import collections

import numpy as np
import pathlib
import re

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from result import y_test_with_patch


def merge_all_files_of_dir(dir):
    list_data = []
    n_patch = 1
    for file in sorted(pathlib.Path(dir).rglob('*.npy')):
        data = np.load(str(file))
        fold, patch = re.split('_', str(file.stem))
        _, n_fold = re.split('-', fold)
        _, n_patch = re.split('-', patch)

        for d in data:
            list_data.append(np.append(d, int(n_fold)))
    return list_data, n_patch


def get_samples_with_patch(x, y, index_train_test, n_patch):
    new_x = np.empty(shape=(0, x.shape[1]))
    new_y = np.empty(shape=(0,))

    for index in index_train_test:
        start = (index * n_patch)
        end = start + n_patch
        new_x = np.concatenate([new_x, x[start:end]])
        new_y = np.concatenate([new_y, y[start:end]])

    return new_x, new_y


def get_cv(cfg, data):
    k = StratifiedKFold(n_splits=cfg['fold'], shuffle=True, random_state=cfg['seed'])
    x_split = np.random.rand(int(data['n_samples']/data['n_patch']), data['n_features'])
    y_split = y_test_with_patch(data['n_patch'], data['y'])
    return k.split(x_split, y_split)


def add_data(color_mode, dataset, dir, extractor, image_size, n_features, n_labels, n_patch, n_samples, segmented,
             slice_patch, x, y):
    return {
        'color_mode': color_mode,
        'dataset': dataset,
        'dir': dir,
        'extractor': extractor,
        'image_size': int(image_size),
        'n_labels': n_labels,
        'n_features': n_features,
        'n_patch': int(n_patch) if n_patch else 1,
        'n_samples': n_samples,
        'segmented': segmented,
        'slice_patch': slice_patch,
        'x': x,
        'y': y
    }


def search_info(list_info, info):
    result = [i for i in list_info if i in str(info).lower()]
    return check_has_result(result)


def check_has_result(result):
    return None if len(result) == 0 else result[0]


def get_info(path):
    dataset = search_info(['george', 'sp', 'specieslink', 'br', 'brasil', 'regioes', 'iwssip'], str(path))
    color_mode = search_info(['grayscale', 'rgb'], str(path))
    segmented = search_info(['manual', 'unet'], str(path))
    dim = search_info(['256', '400', '512'], str(path))
    extractor = search_info(['lbp', 'surf64', 'surf128', 'mobilenetv2', 'resnet50v2', 'vgg16'], str(path))
    slice_patch = search_info(['horizontal', 'vertical', 'h+v'], str(path))

    return dataset, color_mode, segmented, dim, extractor, slice_patch


def show_info_data(data):
    print(f'[INFO] dataset: {data["dataset"]} color_mode: {data["color_mode"]}')
    print(f'[INFO] segmented: {data["segmented"]} image_size: {data["image_size"]} extractor: {data["extractor"]}')
    print(f'[INFO] n_samples/patch: {int(data["n_samples"]) / int(data["n_patch"])}')
    print(f'[INFO] n_samples: {data["n_samples"]} n_features: {data["n_features"]}')
    print(f'[INFO] n_labels: {data["n_labels"]} samples_per_labels: {collections.Counter(data["y"])}')


def show_info_data_train_test(classifier_name, fold, x_test, x_train, y_test, y_train):
    print(fold, classifier_name, x_train.shape, x_test.shape)
    print('[TRAIN]' + str(sorted(list(collections.Counter(y_train).items()))))
    print('[TEST]' + str(sorted(list(collections.Counter(y_test).items()))))


def data_with_pca(cfg, color_mode, d, dataset, extractor, image_size, list_data, list_extractor, n_features, n_labels, n_patch, n_samples, segmented, slice_patch, x_normalized, y):

    for pca in list_extractor[extractor]:
        if pca_is_less_than_n_features(n_features, pca):
            x = PCA(n_components=pca, random_state=cfg['seed']).fit_transform(x_normalized)
            list_data.append(
                add_data(color_mode, dataset, d, extractor, image_size, x.shape[1], n_labels, n_patch, n_samples,
                         segmented, slice_patch, x, y))


def pca_is_less_than_n_features(n_features, pca):
    return pca < n_features - 1


def data_contains_nan(x):
    return np.isnan(x).any()


def get_x_y(cfg, color_mode, data, dataset, extractor, file, image_size, list_data, list_extractor, n_patch, segmented,
            slice_patch):
    n_samples, n_features = data.shape
    x, y = data[0:, 0:n_features - 1], data[:, n_features - 1]
    n_labels = len(np.unique(y))
    x_normalized = StandardScaler().fit_transform(x)

    # if data_contains_nan(x_normalized):
    #     raise ValueError(f'data contain nan')

    list_data.append(add_data(color_mode, dataset, file, extractor, image_size, n_features - 1, n_labels, n_patch,
                              n_samples, segmented, slice_patch, x_normalized, y))
    data_with_pca(cfg, color_mode, file, dataset, extractor, image_size, list_data, list_extractor, n_features,
                  n_labels, n_patch, n_samples, segmented, slice_patch, x_normalized, y)


def split_train_test(data, index_test, index_train, handcraft=False):
    return check_split_train_test(data, handcraft, index_test, index_train)


def check_split_train_test(data, handcraft, index_test, index_train):
    """

    :param data:
    :param handcraft:
    :param index_test:
    :param index_train:
    :return: x_test, x_train, y_test, y_train
    """
    return split_train_test_handcraft(data, index_test, index_train) if handcraft else split_train_test_non_handcraft(data, index_test, index_train)


def split_train_test_non_handcraft(data, index_test, index_train):
    x_train, y_train = get_samples_with_patch(data['x'], data['y'], index_train, data['n_patch'])
    x_test, y_test = get_samples_with_patch(data['x'], data['y'], index_test, data['n_patch'])
    return x_test, x_train, y_test, y_train


def split_train_test_handcraft(data, index_test, index_train):
    x = data['x']
    y = data['y']
    x_train, y_train = x[index_train], y[index_train]
    x_test, y_test = x[index_test], y[index_test]
    return x_test, x_train, y_test, y_train


def load_data(cfg, list_extractor, list_inputs, handcraft=False):
    return check_data(cfg, handcraft, list_extractor, list_inputs)


def check_data(cfg, handcraft, list_extractor, list_inputs):
    return data_is_a_file_handcraft(cfg, list_extractor, list_inputs) if handcraft else data_is_a_dir_non_handcraft(cfg, list_extractor, list_inputs)


def data_is_a_dir_non_handcraft(cfg, list_extractor, list_inputs):
    list_data = []
    for d in list_inputs:
        dataset, color_mode, segmented, image_size, extractor, slice_patch = get_info(d)
        data, n_patch = merge_all_files_of_dir(d)
        get_x_y(cfg, color_mode, np.array(data), dataset, extractor, d, image_size, list_data, list_extractor, n_patch,
                segmented, slice_patch)
    return list_data


def data_is_a_file_handcraft(cfg, list_extractor, list_inputs):
    list_data = []
    for file in list_inputs:
        dataset, color_mode, segmented, image_size, extractor, slice_patch = get_info(file)
        get_x_y(cfg, color_mode, np.loadtxt(file), dataset, extractor, file, image_size, list_data, list_extractor, 1,
                segmented, slice_patch)
    return list_data
