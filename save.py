import collections
import itertools
import math
import pathlib
from typing import LiteralString

import joblib
import logging
import numpy as np
import os
import pandas as pd

from config import Config
from dataset import Dataset

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)

extractors = ['mobilenetv2', 'vgg16', 'resnet50v2', 'lbp', 'surf64', 'surf128']
image_size = [256, 400, 512]
classifiers_name = [
    'KNeighborsClassifier',
    'MLPClassifier',
    'RandomForestClassifier',
    'SVC',
    'DecisionTreeClassifier',
]
columns = ['%s+%s' % (name, image) for name in classifiers_name for image in image_size]
dimensions = {
    'mobilenetv2': [1280, 1024, 512, 256, 128],
    'vgg16': [512, 256, 128],
    'resnet50v2': [2048, 1024, 512, 256, 128],
    'lbp': [59],
    'surf64': [257, 256, 128],
    'surf128': [513, 512, 256, 128]
}
index = ['%s+%s+%s' % (extractor, dimension, metric) for extractor in extractors for dimension in dimensions[extractor]
         for metric in ['mean', 'std']]


def save_means(config: Config, means: list, output: pathlib.Path | LiteralString | str):
    output = os.path.join(output, 'mean')
    os.makedirs(output, exist_ok=True)
    filename = os.path.join(output, 'means.csv')

    logging.info('Saving %s' % filename)
    data = {'mean': [], 'metric': [], 'std': [], 'rule': []}
    df = pd.DataFrame(data, columns=data.keys())
    for mean in means:
        df = pd.concat([df, mean.save()], axis=0)
    df.to_csv(filename, index=False, header=True, sep=';', quoting=2, encoding='utf-8')


def save_folds(config: Config, dataset: Dataset, folds: list, output: pathlib.Path | LiteralString | str):
    for fold in folds:
        fold.save(dataset.levels, output)


def save(config: Config, dataset: Dataset, folds: list, means: list, output: pathlib.Path | LiteralString | str):
    save_folds(config, dataset, folds, output)
    save_means(config, means, output)
