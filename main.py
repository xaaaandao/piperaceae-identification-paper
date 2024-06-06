import dataclasses
import inspect
import pathlib
import sys

import click
import datetime
import logging
import os.path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# import config
from classifiers import get_classifiers, select_classifiers
from config import Config
from dataset import Dataset
from image import Image
from fold import Fold
from mean import Mean
from save import save

# from dataset import load_dataset, split_folds

FOLDS = 5
GPU_ID = 0
N_JOBS = -1
SEED = 1234
OUTPUT = '/home/none/results'
VERBOSE = 42

datefmt = '%d-%m-%Y+%H-%M-%S'
dateandtime = datetime.datetime.now().strftime(datefmt)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)

parameters = {
    'DecisionTreeClassifier': {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [10, 100, 1000]
    },
    'KNeighborsClassifier': {
        'n_neighbors': [2, 4, 6, 8, 10],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'MLPClassifier': {
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['adam', 'sgd'],
        'learning_rate_init': [0.01, 0.001, 0.0001],
        'momentum': [0.9, 0.4, 0.1]
    },
    'RandomForestClassifier': {
        'n_estimators': [200, 400, 600],
        'max_features': ['sqrt', 'log2'],
        'criterion': ['gini', 'entropy']
    },
    'SVC': {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }
}


def has_pca(dataset: Dataset, extractors: dict, x: np.ndarray) -> list:
    return [PCA(n_components=d, random_state=SEED).fit_transform(x) for d in extractors[dataset.model.lower()] if
            d < dataset.features]


def apply_pca(dataset: Dataset, extractors: dict, pca: bool, x: np.ndarray) -> list:
    return has_pca(dataset, extractors, x) if pca and dataset.model else [x]


class Extractors:
    extractors: dict = dataclasses.field(default_factory={
        'mobilenetv2': [1280, 1024, 512, 256, 128],
        'vgg16': [512, 256, 128],
        'resnet50v2': [2048, 1024, 512, 256, 128],
        'lbp': [59],
        'surf64': [257, 256, 128],
        'surf128': [513, 512, 256, 128]
    })


def get_output_name(classifier_name: str, dataset: Dataset) -> str:
    path = ''
    for k, v in dataset.__dict__.items():
        print(k, v, type(v))
        if v and not isinstance(v, list) and not isinstance(v, Image) and 'input' not in k:
            path = path + '%s+%s-' % (k, str(v))
    return path + 'clf+%s' % classifier_name


@click.command()
@click.option('-C', '--config', type=click.types.Path(exists=True), required=False)
@click.option('-c', '--clf', multiple=True, type=click.Choice(get_classifiers()),
              default=['DecisionTreeClassifier'])
@click.option('-i', '--input', required=True)
@click.option('-o', '--output', required=False, default='output')
@click.option('-p', '--pca', is_flag=True, default=False)
def main(config, clf, input, output, pca):
    config = Config()
    config._print()
    classifiers = select_classifiers(config, clf)
    extractors = Extractors()

    if len(classifiers) == 0:
        raise SystemExit('classifiers choosed not found')

    if not os.path.exists(input):
        raise SystemExit('input %s not found' % input)

    dataset = Dataset(input)
    dataset.load()
    dataset._print()

    x, y = dataset.load_features()
    index = dataset.split_folds(config, y)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    if np.isnan(x).any():
        logging.error('x contains NaN values')
        raise ValueError

    xs = apply_pca(dataset, extractors.extractors, pca, x)

    results = []
    for x in xs:
        for c in classifiers:
            classifier_name = c.__class__.__name__
            logging.info('the current classifier is %s' % classifier_name)

            dataset.count_features = x.shape[1]
            ran = list(
                [pathlib.Path(output).rglob('*%s*' + dataset.get_output_name(classifier_name, dataset.count_features))])
            # this program create xxx files csv
            # if len(ran) > 0 and any(len(list(pathlib.Path(r).rglob('*.csv'))) == 10 for r in ran):
            #     logging.info('')
            #     sys.exit(1)

            # TODO falta a pasta, e validar se já rodou
            output = os.path.join(output, dataset.get_output_name(classifier_name, dataset.count_features))
            os.makedirs(output, exist_ok=True)

            clf = GridSearchCV(c, parameters[classifier_name], cv=config.folds, scoring=config.cv_metric,
                               n_jobs=config.n_jobs, verbose=config.verbose)

            with joblib.parallel_backend(config.backend, n_jobs=config.n_jobs):
                clf.fit(x, y)

            # enable to use predict_proba
            if isinstance(clf.best_estimator_, SVC):
                params = dict(probability=True)
                clf.best_estimator_.set_params(**params)

            folds = []
            for f, idx in enumerate(index, start=1):
                fold = Fold(f, idx, x, y)
                fold.run(clf, dataset)
                folds.append(fold)

            means = [Mean(folds, dataset.levels, 'max'),
                     Mean(folds, dataset.levels, 'sum'),
                     Mean(folds, dataset.levels, 'mult')]
            save(clf, config, dataset, folds, means, output)


if __name__ == '__main__':
    main()
