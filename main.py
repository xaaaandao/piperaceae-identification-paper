import dataclasses

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
from fold import Fold
from result import Mean

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
    extractors : dict = dataclasses.field(default_factory={
        'mobilenetv2': [1280, 1024, 512, 256, 128],
        'vgg16': [512, 256, 128],
        'resnet50v2': [2048, 1024, 512, 256, 128],
        'lbp': [59],
        'surf64': [257, 256, 128],
        'surf128': [513, 512, 256, 128]
    })



@click.command()
@click.option('-C', '--config', type=click.types.Path(exists=True), required=False)
@click.option('-c', '--clf', multiple=True, type=click.Choice(get_classifiers()),
              default=['DecisionTreeClassifier'])
@click.option('-i', '--input', required=True)
@click.option('-p', '--pca', is_flag=True, default=False)
def main(config, clf, input, pca):
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

            # TODO falta a pasta, e validar se já rodou

            clf = GridSearchCV(c, parameters[classifier_name], cv=config.folds, scoring=config.metric, n_jobs=config.n_jobs, verbose=config.verbose)

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
            mean = [Mean(folds, 'max'),
                    Mean(folds, 'sum'),
                    Mean(folds, 'mult')]
                # results_fold.append(result)
    #
    #             logging.info('results_fold %s' % str(len(results_fold)))
    #             means = mean_metrics(results_fold, n_labels)
    #             save_mean(means, path, results_fold)
    #             save_best(clf, means, path, results_fold)
    #             save_info(classifier_name, extractor, n_features, n_samples, path, patch)
    #             list_results_classifiers.append({
    #                 'classifier_name': classifier_name,
    #                 'image_size': str(image_size[0]),
    #                 'extractor': extractor,
    #                 'n_features': str(n_features),
    #                 'means': means
    #             })
    #             save_df_main(color, dataset, minimum_image, list_results_classifiers, output_base, region=region)


if __name__ == '__main__':
    main()
