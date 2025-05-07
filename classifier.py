import multiprocessing
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

cfg_classifier = {
    'n_jobs': -1,
    'seed': 1234
}

list_params = {
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
        'n_estimators': [200, 400, 600, 800, 1000],
        'max_features': ['sqrt', 'log2'],
        'criterion': ['gini', 'entropy'],
        'max_depth': [10, 100, 1000]
    },
    'SVC': {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    }
}

list_classifiers = [
    # DecisionTreeClassifier(random_state=cfg_classifier['seed']),
    # KNeighborsClassifier(n_jobs=cfg_classifier['n_jobs']),
    # MLPClassifier(random_state=cfg_classifier['seed']),
    RandomForestClassifier(random_state=cfg_classifier['seed'], n_jobs=cfg_classifier['n_jobs']),
    # SVC(random_state=1234, verbose=True, probability=True)
]


def find_best_classifier_and_params(cfg, classifier, data, metric):
    classifier_name = classifier.__class__.__name__

    print(f'[GRIDSEARCH CV] find best params of {classifier_name}')

    classifier_best_params = GridSearchCV(classifier, list_params[classifier_name], scoring=metric, cv=cfg['fold'], pre_dispatch=int(multiprocessing.cpu_count()), verbose=cfg['verbose'])

    start_search_best_params = time.time()
    classifier_best_params.fit(data['x'], data['y'])
    end_search_best_params = time.time()
    time_search_best_params = end_search_best_params - start_search_best_params

    best_classifier = classifier_best_params.best_estimator_
    best_params = classifier_best_params.best_params_

    return {'classifier': best_classifier, 'params': best_params}, classifier_name, time_search_best_params

