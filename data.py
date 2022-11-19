import collections
import math

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
    # result = list(filter(lambda x: x in str(info).lower(), list_info))
    result = [i for i in list_info if i in str(info).lower()]
    return None if len(result) == 0 else result[0]


def get_info(path):
    dataset = search_info(['george', 'sp', 'specieslink'], str(path))
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
    # print('train')
    print('[TRAIN]' + str(sorted(list(collections.Counter(y_train).items()))))
    # print('test')
    print('[TEST]' + str(sorted(list(collections.Counter(y_test).items()))))


def data_with_pca(cfg, color_mode, d, dataset, extractor, image_size, list_data, list_extractor, n_features, n_labels, n_patch, n_samples, segmented, slice_patch, x_normalized, y):

    for pca in list_extractor[extractor]:
        if pca < n_features - 1:
            x = PCA(n_components=pca, random_state=cfg['seed']).fit_transform(x_normalized)
            list_data.append(
                add_data(color_mode, dataset, d, extractor, image_size, x.shape[1], n_labels, n_patch, n_samples,
                         segmented, slice_patch, x, y))


def get_x_y(cfg, color_mode, data, dataset, extractor, file, image_size, list_data, list_extractor, n_patch, segmented,
            slice_patch):
    n_samples, n_features = data.shape
    x, y = data[0:, 0:n_features - 1], data[:, n_features - 1]
    if np.isnan(x).any():
        raise ValueError(f'data contain nan')
    n_labels = len(np.unique(y))
    x_normalized = StandardScaler().fit_transform(x)
    list_data.append(add_data(color_mode, dataset, file, extractor, image_size, n_features - 1, n_labels, n_patch,
                              n_samples, segmented, slice_patch, x_normalized, y))
    data_with_pca(cfg, color_mode, file, dataset, extractor, image_size, list_data, list_extractor, n_features,
                  n_labels, n_patch, n_samples, segmented, slice_patch, x_normalized, y)
