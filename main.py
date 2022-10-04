import datetime
import pathlib
import re
import time

import joblib
import numpy as np
import os

import sklearn.ensemble
import sklearn.model_selection
import sklearn.neighbors
import sklearn.neural_network
import sklearn.svm
import sklearn.tree

from result import calculate_test
from samples import get_samples_with_patch
from save import save_mean, save_fold, save_info_dataset


def main():
    cfg = {
        'fold': 5,
        'n_jobs': -1,
        'n_samples': 375,
        'n_labels': 5,
        'seed': 1234,
        'dir_input': '../dataset/features',
        'dir_output': 'out'
    }

    list_extractor = {
        'lbp': [59],
        'surf64': [128, 256, 257],
        'surf128': [128, 256, 512, 513],
        'mobilenetv2': [128, 256, 512, 1024, 1280],
        'resnet50v2': [128, 256, 512, 1024, 2048],
        'vgg16': [128, 256, 512]
    }

    list_classifiers = [
        sklearn.tree.DecisionTreeClassifier(random_state=cfg['seed']),
        sklearn.neighbors.KNeighborsClassifier(n_jobs=cfg['n_jobs']),
        sklearn.neural_network.MLPClassifier(random_state=cfg['seed']),
        sklearn.ensemble.RandomForestClassifier(random_state=cfg['seed'], n_jobs=cfg['n_jobs']),
        sklearn.svm.SVC(random_state=cfg['seed'], probability=True)
    ]

    list_hyperparametrs = {
        "DecisionTreeClassifier": {
            "criterion": ["gini", "entropy"],
            "splitter": ["best", "random"],
            "max_depth": [10, 100, 1000]
        },
        "KNeighborsClassifier": {
            "n_neighbors": [2, 4, 6, 8, 10],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        },
        "MLPClassifier": {
            "activation": ["identity", "logistic", "tanh", "relu"],
            "solver": ["adam", "sgd"],
            "learning_rate_init": [0.01, 0.001, 0.0001],
            "momentum": [0.9, 0.4, 0.1]
        },
        "RandomForestClassifier": {
            "n_estimators": [200, 400, 600, 800, 1000],
            "max_features": ["sqrt", "log2"],
            "criterion": ["gini", "entropy"],
            "max_depth": [10, 100, 1000]
        },
        "SVC": {
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
        }
    }

    current_datetime = datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
    kf = sklearn.model_selection.KFold(n_splits=cfg['fold'], shuffle=True, random_state=cfg['seed'])
    list_data_input = [
        os.path.join(cfg['dir_input'], 'manual', 'RGB', '256', 'mobilenetv2', 'horizontal', 'patch=3')
    ]

    list_only_file = [file for file in list_data_input if os.path.isfile(file)]
    for file in list_only_file:
        _, _, _, dataset, color_mode, dim, filename = re.split('/', file)
        data = np.loadtxt(file)
        n_samples, n_features = data.shape
        x, y = data[0:, 0:n_features - 1], data[:, n_features - 1]
        x_normalized = sklearn.preprocessing.StandardScaler().fit_transform(x)

        extractor = filename.replace('.txt', '')
        slice = None
        n_patch = None

        list_data_pca = p(cfg, extractor, list_extractor, x_normalized, y)

        for data in list_data_pca:
            for classifier in list_classifiers:
                classifier_name = classifier.__class__.__name__

                classifier_best_params = sklearn.model_selection.GridSearchCV(classifier,
                                                                              list_hyperparametrs[classifier_name],
                                                                              scoring='accuracy', cv=cfg['fold'],
                                                                              verbose=42, n_jobs=cfg['n_jobs'])

                start_search_best_hyperparameters = time.time()
                classifier_best_params.fit(data['x'], data['y'])
                end_search_best_hyperparameters = time.time()
                time_search_best_params = end_search_best_hyperparameters - start_search_best_hyperparameters

                best_classifier = classifier_best_params.best_estimator_
                best_params = classifier_best_params.best_params_

                list_result_fold = []
                list_time = []

                path = os.path.join(cfg['dir_output'], current_datetime, dataset, dim, extractor, classifier_name, f'patch=None',
                                    str(data['pca']))
                pathlib.Path(path).mkdir(parents=True, exist_ok=True)

                for fold, (index_train, index_test) in enumerate(kf.split(np.random.rand(cfg['n_samples'], ))):
                    x_train, y_train = x[index_train], y[index_train]
                    x_test, y_test = x[index_test], y[index_test]

                    start_time_train_valid = time.time()
                    best_classifier.fit(x_train, y_train)
                    y_pred = best_classifier.predict_proba(x_test)

                    result_max_rule, result_prod_rule, result_sum_rule = calculate_test(cfg, fold, y_pred, y_test)
                    end_time_train_valid = time.time()
                    time_train_valid = end_time_train_valid - start_time_train_valid

                    list_result_fold.append(result_max_rule)
                    list_result_fold.append(result_prod_rule)
                    list_result_fold.append(result_sum_rule)
                    list_time.append({
                        "fold": fold,
                        "time_train_valid": time_train_valid,
                        "time_search_best_params": time_search_best_params
                    })

                save_fold(cfg, classifier_name, dataset, list_result_fold, list_time, path)
                save_mean(best_params, list_result_fold, list_time, path)
                save_info_dataset(color_mode, data, dataset, dim, file, extractor, n_patch, path, slice)

    list_only_dir = [dir for dir in list_data_input if os.path.isdir(dir)]
    for dir in list_only_dir:
        _, _, _, dataset, color_mode, dim, extractor, slice, _ = re.split('/', dir)
        print(dataset, color_mode, dim, extractor, slice)
        list_data = []
        for file in sorted(pathlib.Path(dir).rglob('*.npy')):
            data = np.load(str(file))
            fold, patch = re.split('_', str(file.stem))
            _, n_fold = re.split('-', fold)
            _, n_patch = re.split('-', patch)

            for d in data:
                list_data.append(np.append(d, int(n_fold)))

        new_data = np.array(list_data)
        n_samples, n_features = new_data.shape
        x, y = new_data[0:, 0:n_features - 1], new_data[:, n_features - 1]
        x_normalized = sklearn.preprocessing.StandardScaler().fit_transform(x)

        list_data_pca = p(cfg, extractor, list_extractor, x_normalized, y)

        for data in list_data_pca:
            for classifier in list_classifiers:
                classifier_name = classifier.__class__.__name__

                classifier_best_params = sklearn.model_selection.GridSearchCV(classifier, list_hyperparametrs[classifier_name],
                                                              scoring='accuracy', cv=cfg['fold'],
                                                              verbose=42, n_jobs=cfg['n_jobs'])

                start_search_best_hyperparameters = time.time()
                classifier_best_params.fit(data['x'], data['y'])
                end_search_best_hyperparameters = time.time()
                time_search_best_params = end_search_best_hyperparameters - start_search_best_hyperparameters

                best_classifier = classifier_best_params.best_estimator_
                best_params = classifier_best_params.best_params_

                list_result_fold = []
                list_time = []

                path = os.path.join(cfg['dir_output'], current_datetime, dataset, dim, extractor, classifier_name,
                                    f'patch={n_patch}', str(data['pca']))
                pathlib.Path(path).mkdir(parents=True, exist_ok=True)

                for fold, (index_train, index_test) in enumerate(kf.split(np.random.rand(cfg['n_samples'], ))):
                    x_train, y_train = get_samples_with_patch(data['x'], data['y'], index_train, int(n_patch))
                    x_test, y_test = get_samples_with_patch(data['x'], data['y'], index_test, int(n_patch))

                    print(fold, classifier_name, x_train.shape, x_test.shape)
                    start_time_train_valid = time.time()
                    best_classifier.fit(x_train, y_train)
                    y_pred = best_classifier.predict_proba(x_test)

                    result_max_rule, result_prod_rule, result_sum_rule = calculate_test(cfg, fold, y_pred, y_test, n_patch=int(n_patch))
                    end_time_train_valid = time.time()
                    time_train_valid = end_time_train_valid - start_time_train_valid

                    list_result_fold.append(result_max_rule)
                    list_result_fold.append(result_prod_rule)
                    list_result_fold.append(result_sum_rule)
                    list_time.append({
                        "fold": fold,
                        "time_train_valid": time_train_valid,
                        "time_search_best_params": time_search_best_params
                    })

                save_fold(cfg, classifier_name, dataset, list_result_fold, list_time, path)
                save_mean(best_params, list_result_fold, list_time, path)
                save_info_dataset(color_mode, data, dataset, dim, dir, extractor, n_patch, path, slice)


def p(cfg, extractor, list_extractor, x_normalized, y):
    list_data_pca = []
    for pca in list_extractor[extractor]:
        list_data_pca.append({
            'x': x_normalized if pca == max(list_extractor[extractor]) else sklearn.decomposition.PCA(
                n_components=pca, random_state=cfg['seed']).fit_transform(x_normalized),
            'y': y,
            'pca': pca
        })
    return list_data_pca


if __name__ == '__main__':
    main()


