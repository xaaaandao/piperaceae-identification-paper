import collections

import numpy as np
import pathlib
import re

import sklearn.decomposition
import sklearn.model_selection


def merge_all_files_of_dir(dir):
    list_data = []
    n_patch = -1
    for file in sorted(pathlib.Path(dir).rglob('*.npy')):
        data = np.load(str(file))
        fold, patch = re.split('_', str(file.stem))
        _, n_fold = re.split('-', fold)
        _, n_patch = re.split('-', patch)

        for d in data:
            list_data.append(np.append(d, int(n_fold)))
    return list_data, n_patch


def get_samples_with_patch(x, y, list_index, n_patch):
    new_x = np.zeros(shape=(0, x.shape[1]))
    new_y = np.zeros(shape=(0,))

    for index in list_index:
        start = (index * n_patch)
        end = start + n_patch
        new_x = np.concatenate([new_x, x[start:end]])
        new_y = np.concatenate([new_y, y[start:end]])

    return new_x, new_y


def get_cv(cfg, data):
    k = sklearn.model_selection.StratifiedKFold(n_splits=cfg['fold'], shuffle=True,
                                                random_state=cfg['seed'])
    return k.split(data['x'], data['y'])


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
        'n_patch': int(n_patch) if n_patch else -1,
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
    extractor = search_info(['lbp', 'surf', 'mobilenetv2', 'resnet50v2', 'vgg16'], str(path))
    slice_patch = search_info(['horizontal', 'vertical', 'h+v'], str(path))

    return dataset, color_mode, segmented, dim, extractor, slice_patch


def show_info_data(data):
    print(f'dataset: {data["dataset"]} color_mode: {data["color_mode"]}')
    print(f'segmented: {data["segmented"]} image_size: {data["image_size"]} extractor: {data["extractor"]}')
    print(f'n_samples/patch: {int(data["n_samples"]) / int(data["n_patch"])}')
    print(f'n_samples: {data["n_samples"]} n_features: {data["n_features"]}')
    print(f'n_labels: {data["n_labels"]} samples_per_labels: {collections.Counter(data["y"])}')


def show_info_data_train_test(classifier_name, fold, x_test, x_train, y_test, y_train):
    print(fold, classifier_name, x_train.shape, x_test.shape)
    print('train')
    print(sorted(list(collections.Counter(y_train).items())))
    print('test')
    print(sorted(list(collections.Counter(y_test).items())))


def data_with_pca(cfg, color_mode, d, dataset, extractor, image_size, list_data, list_extractor, n_features, n_labels, n_patch, n_samples, segmented, slice_patch, x_normalized, y):
    for pca in list_extractor[extractor]:
        if pca < n_features - 1:
            x = sklearn.decomposition.PCA(n_components=pca, random_state=cfg['seed']).fit_transform(x_normalized)
            list_data.append(
                add_data(color_mode, dataset, d, extractor, image_size, x.shape[1], n_labels, n_patch, n_samples,
                         segmented, slice_patch, x, y))
