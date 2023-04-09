import collections
import itertools
import logging
import os
import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def load_dataset_informations(input):
    p = pathlib.Path(input)
    if p.is_file() and os.path.exists(str(p).replace(str(p.name), 'info.csv')):
        file_with_info = str(p).replace(str(p.name), 'info.csv')
        df = pd.read_csv(file_with_info, sep=',', header=0)
        query = 'extractor==\'%s\'' %  str(p.stem)
        color = df.query(query)['color'][0]
        dataset = df.query(query)['dataset'][0]
        extractor = df.query(query)['extractor'][0]
        height = int(df.query(query)['height'][0])
        info_dataset = df.query(query)['dataset'][0]
        input_path = df.query(query)['input_path'][0]
        minimum_image = df.query(query)['minimum_image'][0]
        n_features = int(df.query(query)['n_features'][0])
        n_samples = int(df.query(query)['total_samples'][0])
        patch = 1
        width = int(df.query(query)['width'][0])
    else:
        color, dataset, extractor, height, info_dataset, input_path, minimum_image, n_features, n_samples, patch, width = information_about_dataset(input)

    input_path = input_path.replace('_features', '')
    print(input_path)
    if not os.path.exists(input_path):
        raise SystemExit('input path %s not exists' % input_path)

    list_info_level = information_about_level(info_dataset, input, input_path)

    return color, dataset, extractor, (height, width), list_info_level, minimum_image, n_features, n_samples, patch


def information_about_dataset(input):
    info_dataset = [f for f in pathlib.Path(input).rglob('info.csv') if f.is_file()]
    if len(info_dataset) == 0:
        raise SystemExit('info.csv not found in %s' % input)
    df = pd.read_csv(info_dataset[0], index_col=0, header=None, sep=';')
    extractor = df.loc['cnn'][1]
    color = df.loc['color'][1]
    dataset = df.loc['dataset'][1]
    input_path = df.loc['input_path'][1]
    minimum_image = int(df.loc['minimum_image'][1])
    n_features = int(df.loc['n_features'][1])
    n_samples = int(df.loc['total_samples'][1])
    height = int(df.loc['height'][1])
    width = int(df.loc['width'][1])
    patch = int(df.loc['patch'][1])
    logging.info('[INFO] n_samples: %s n_features: %s patch: %s' % (n_samples, n_features, patch))
    return color, dataset, extractor, height, info_dataset, input_path, minimum_image, n_features, n_samples, patch, width


def information_about_level(info_dataset, input, input_path):
    info_level = [f for f in pathlib.Path(input_path).rglob('info_levels.csv') if f.is_file()]
    if len(info_dataset) == 0:
        raise SystemExit('info_levels.csv not found in %s' % input)
    logging.info('[INFO] reading file %s' % str(info_level[0]))
    df = pd.read_csv(info_level[0], index_col=None, usecols=['levels', 'count', 'f'], sep=';')
    list_info_level = df[['levels', 'count', 'f']].to_dict()
    logging.info('[INFO] n_levels: %s' % str(len(list_info_level['levels'])))
    return list_info_level


def prepare_data_txt(input):
    data = np.loadtxt(input)
    n_samples, n_features = data.shape
    x, y = data[0:, 0:n_features - 1], data[:, n_features - 1]
    return x, y


def prepare_data(folds, input, n_features, n_samples, patch, seed):
    if patch > 0 and input.endswith('.txt') and os.path.isfile(input):
        x, y = prepare_data_txt(input)
    else:
        x, y = prepare_data_npz(input, n_features)

    logging.info('[INFO] dataset contains x.shape: %s' % str(x.shape))
    logging.info('[INFO] dataset contains y.shape: %s' % str(y.shape))

    index = split_folds(folds, n_features, n_samples, patch, seed, y)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return index, x, y


def prepare_data_npz(input, n_features):
    x = np.empty(shape=(0, n_features), dtype=np.float64)
    y = []
    for fname in sorted(pathlib.Path(input).rglob('*.npz')):
        if fname.is_file():
            d = np.load(fname)
            x = np.append(x, d['x'], axis=0)
            y.append(d['y'])
    y = np.array(list(itertools.chain(*y)), dtype=np.int16)
    return x, y


def split_folds(folds, n_features, n_samples, patch, seed, y):
    np.random.seed(seed)
    x = np.random.rand(int(n_samples / patch), n_features)
    y = [np.repeat(k, int(v / patch)) for k, v in dict(collections.Counter(y)).items()]
    y = np.array(list(itertools.chain(*y)))
    logging.info('[INFO] StratifiedKFold x.shape: %s' % str(x.shape))
    logging.info('[INFO] StratifiedKFold y.shape: %s' % str(y.shape))
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return kf.split(x, y)

