import collections

import numpy as np
import pathlib
import re
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
    samples_per_label = [v for _, v in collections.Counter(data['y']).items()]
    return dataset_is_balanced(cfg, data) if all(e == samples_per_label[0] for e in samples_per_label) \
        else dataset_is_unbalanced(cfg, data)


def dataset_is_unbalanced(cfg, data):
    print('unbalanced dataset')
    k = sklearn.model_selection.StratifiedKFold(n_splits=cfg['fold'], shuffle=True,
                                                random_state=cfg['seed'])
    return k.split(data['x'], data['y'])


def dataset_is_balanced(cfg, data):
    print('balanced dataset')
    k = sklearn.model_selection.KFold(n_splits=cfg['fold'], shuffle=True, random_state=cfg['seed'])
    return k.split(data['x'])


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
        'n_patch': int(n_patch),
        'n_samples': n_samples,
        'segmented': segmented,
        'slice_patch': slice_patch,
        'x': x,
        'y': y
    }


def search_info(list_info, info):
    result = list(filter(lambda x: x in str(info).lower(), list_info))
    return None if len(result) == 0 else result[0]


def get_info(path):
    dataset = search_info(['george', 'sp', 'specieslink'], str(path))
    color_mode = search_info(['grayscale', 'rgb'], str(path))
    segmented = search_info(['manual', 'unet'], str(path))
    dim = search_info(['256', '400', '512'], str(path))
    extractor = search_info(['lbp', 'surf', 'mobilenetv2', 'resnet50v2', 'vgg16'], str(path))
    slice_patch = search_info(['horizontal', 'vertical', 'h+v'], str(path))

    return dataset, color_mode, segmented, dim, extractor, slice_patch
