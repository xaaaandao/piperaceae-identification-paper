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

from classifiers import get_classifiers, select_classifiers
from config import Config
from dataset import Dataset
from image import Image
from fold import Fold
from mean import Mean
from features import Features
from save import save


datefmt = '%d-%m-%Y+%H-%M-%S'
dateandtime = datetime.datetime.now().strftime(datefmt)
logging.basicConfig(format='\033[32m [%(asctime)s] (%(levelname)s) {%(filename)s %(lineno)d}  %(message)s \033[0m', datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)
logging.basicConfig(format='\033[31m [%(asctime)s] (%(levelname)s) {%(filename)s %(lineno)d}  %(message)s \033[0m', datefmt='%d/%m/%Y %H:%M:%S', level=logging.WARNING)
logging.basicConfig(format='\033[35m [%(asctime)s] (%(levelname)s) {%(filename)s %(lineno)d}  %(message)s \033[0m', datefmt='%d/%m/%Y %H:%M:%S', level=logging.CRITICAL)

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

def has_pca(config: Config, dataset: Dataset, extractors: dict, x: np.ndarray) -> list:
    """
    Aplica PCA no conjunto de dados. Os valores estão definidos em um dicionário.
    :param config: classe config com os valores das configurações dos experimentos.
    :param dataset: classe dataset com informações do conjunto de dados.
    :param extractors: dicionário com os extratores
    :param x: matriz com as features.
    :return: list, lista com as features reduzidas.
    """
    return [PCA(n_components=d, random_state=config.seed).fit_transform(x) for d in extractors[dataset.model.lower()] if d < dataset.features]


def apply_pca(config: Config, dataset: Dataset, extractors: dict, pca: bool, x: np.ndarray) -> list:
    """
    Verifica se é necessário aplicar o PCA.
    :param config: classe config com os valores das configurações dos experimentos.
    :param dataset: classe dataset com informações do conjunto de dados.
    :param extractors: dicionário com os extratores
    :param pca: booleano que indica se é necessário aplicar ou não o PCA.
    :param x: matriz com as features.
    :return: list, lista com as features reduzidas.
    """
    return has_pca(config, dataset, extractors, x) if pca and dataset.model else [x]


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
    model = Features()

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
        # raise ValueError
        return

    xs = apply_pca(config, dataset, model.features, pca, x)

    results = []
    for x in xs:
        for c in classifiers:
            classifier_name = c.__class__.__name__
            logging.info('the current classifier is %s' % classifier_name)

            dataset.count_features = x.shape[1]

            target = dataset.get_output_name(classifier_name, dataset.count_features)
            rans = [p.resolve() for p in pathlib.Path(output).rglob('*%s*' % target)]

            # this program create 89 files csv
            # 3 rule sum, max and mult
            if len(dataset.levels) > 0 and len(rans) > 0:
                count_confusion_matrix = len(dataset.levels) * config.folds * 3
                if any(len([p for p in pathlib.Path(r).rglob('*.csv')]) == 89 + count_confusion_matrix for r in rans):
                    logging.warning('the test exist')
                    sys.exit(1)

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
                fold.results(dataset)
                fold.save(dataset, output)
                folds.append(fold)

            means = Mean(folds)
            means.save(output)
            save(clf, config, dataset, folds, output)


if __name__ == '__main__':
    main()
