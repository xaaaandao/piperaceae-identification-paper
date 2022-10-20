import pathlib
import re

import numpy as np
import sklearn.decomposition


def merge_all_files(dir):
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


def data_with_pca(cfg, extractor, list_extractor, x_normalized, y):
    list_data_pca = []
    for pca in list_extractor[extractor]:
        list_data_pca.append({
            'x': x_normalized if pca == max(list_extractor[extractor]) else sklearn.decomposition.PCA(
                n_components=pca, random_state=cfg['seed']).fit_transform(x_normalized),
            'y': y,
            'pca': pca
        })
    return list_data_pca


def get_samples_with_patch(x, y, list_index, n_patch):
    new_x = np.zeros(shape=(0, x.shape[1]))
    new_y = np.zeros(shape=(0,))

    for index in list_index:
        start = (index * n_patch)
        end = start + n_patch
        new_x = np.concatenate([new_x, x[start:end]])
        new_y = np.concatenate([new_y, y[start:end]])

    return new_x, new_y


def search_info(list_info, info):
    result = list(filter(lambda x: x in str(info).lower(), list_info))
    return None if len(result) == 0 else result[0]


def get_info(path):
    dataset = search_info(['george', 'sp', 'specieslink'], str(path))
    color_mode = search_info(['grayscale', 'rgb'], str(path))
    segmented = search_info(['manual', 'unet'], str(path))
    dim = search_info(['256', '400', '512'], str(path))
    extractor = search_info(['lbp', 'surf', 'mobilenetv2', 'resnet50v2', 'vgg16'], str(path))
    slice = search_info(['horizontal', 'vertical', 'h+v'], str(path))


    return dataset, color_mode, segmented, dim, extractor, slice
