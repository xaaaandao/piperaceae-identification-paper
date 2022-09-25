import csv
import datetime
import pathlib
import re
import sys
import time

import joblib
import matplotlib
import numpy as np
import os

import pandas

import classificadores
import extrator
import sklearn.model_selection

from result import calculate_test
from samples import get_samples_with_patch


def main():
    cfg = {
        'fold': 5,
        'n_jobs': -1,
        'n_samples': 375,
        'n_labels': 5,
        'seed': 1234,
        'dir_input': '../new_features',
        'dir_output': 'out'
    }

    list_extractor = {
        'lbp': [59],
        'surf64': [128, 256, 257],
        'surf128': [128, 256, 513],
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
    list_dir_input = [
        os.path.join('../new_features', 'unet/GRAYSCALE/mobilenetv2/256/horizontal/patch=3'),
        os.path.join('../new_features', 'unet/GRAYSCALE/mobilenetv2/256/horizontal/patch=5'),
        os.path.join('../new_features', 'unet/GRAYSCALE/mobilenetv2/256/horizontal/patch=7')
    ]

    for dir_input in list_dir_input:
        _, _, dataset, color_mode, extractor, dim, slice, _ = re.split('/', dir_input)

        list_data = []
        for file in sorted(pathlib.Path(dir_input).rglob('*.npy')):
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

        l_data = []
        for pca in list_extractor[extractor]:
            l_data.append({
                'x': x_normalized if pca == max(list_extractor[extractor]) else sklearn.decomposition.PCA(
                    n_components=pca, random_state=cfg['seed']).fit_transform(x_normalized),
                'y': y,
                'pca': pca
            })

        for data in l_data:
            for classifier in list_classifiers:
                classifier_name = classifier.__class__.__name__

                classifier_best_params = sklearn.model_selection.GridSearchCV(classifier, list_hyperparametrs[classifier_name],
                                                              scoring='accuracy', cv=cfg['fold'],
                                                              verbose=42, n_jobs=cfg['n_jobs'])

                with joblib.parallel_backend('threading', n_jobs=cfg['n_jobs']):
                    start_search_best_hyperparameters = time.time()
                    classifier_best_params.fit(data['x'], data['y'])
                    end_search_best_hyperparameters = time.time()
                time_search_best_params = end_search_best_hyperparameters - start_search_best_hyperparameters

                best_classifier = classifier_best_params.best_estimator_
                best_params = classifier_best_params.best_params_

                list_result_fold = []
                list_time = []

                path = os.path.join(cfg['dir_output'], current_datetime, classifier_name, f'patch={n_patch}', str(data['pca']))
                pathlib.Path(path).mkdir(parents=True, exist_ok=True)

                for fold, (index_train, index_test) in enumerate(kf.split(np.random.rand(375,))):
                    print(len(index_train), len(index_test))
                    x_train, y_train = get_samples_with_patch(data['x'], data['y'], index_train, int(n_patch))
                    x_test, y_test = get_samples_with_patch(data['x'], data['y'], index_test, int(n_patch))

                    start_time_train_valid = time.time()
                    best_classifier.fit(x_train, y_train)
                    y_pred = best_classifier.predict_proba(x_test)

                    result_max_rule, result_prod_rule, result_sum_rule = calculate_test(cfg, fold, y_pred, y_test, n_patch=int(n_patch))
                    end_time_train_valid = time.time()
                    time_train_valid = end_time_train_valid - start_time_train_valid

                    print(result_max_rule['accuracy'], result_sum_rule['accuracy'], result_prod_rule['accuracy'], sep='\n')
                    print('\n')
                    list_result_fold.append(result_max_rule)
                    list_result_fold.append(result_prod_rule)
                    list_result_fold.append(result_sum_rule)
                    list_time.append({
                        "fold": fold,
                        "time_train_valid": time_train_valid,
                        "time_search_best_params": time_search_best_params
                    })

                save_fold(cfg, classifier_name, dataset, list_result_fold, list_time, path)
                save_mean(best_params, list_result_fold, list_time, data['x'].shape[0], path)
                save_info_dataset(color_mode, data, dataset, dim, dir_input, extractor, n_patch, path, slice)
                # break
            # break


def save_mean(best_params, list_result_fold, list_time, n_features, path):
    mean_time = np.mean([t['time_train_valid'] for t in list_time])
    mean_time_millisec = mean_time * 1000
    mean_time_min = mean_time / 60
    mean_time_hour_min_sec = time.strftime('%H:%M:%S', time.gmtime(float(mean_time)))
    std_time = np.std([t['time_train_valid'] for t in list_time])

    list_mean_rule = []
    list_mean_rule.append({
        'mean': np.mean([r['accuracy'] for r in list(filter(lambda x: x['rule'] == 'sum', list_result_fold))]),
        'std': np.std([r['accuracy'] for r in list(filter(lambda x: x['rule'] == 'sum', list_result_fold))]),
        'rule': 'sum'
    })
    list_mean_rule.append({
        'mean': np.mean([r['accuracy'] for r in list(filter(lambda x: x['rule'] == 'max', list_result_fold))]),
        'std': np.std([r['accuracy'] for r in list(filter(lambda x: x['rule'] == 'max', list_result_fold))]),
        'rule': 'max'
    })
    list_mean_rule.append({
        'mean': np.mean([r['accuracy'] for r in list(filter(lambda x: x['rule'] == 'prod', list_result_fold))]),
        'std': np.std([r['accuracy'] for r in list(filter(lambda x: x['rule'] == 'prod', list_result_fold))]),
        'rule': 'prod'
    })

    best_mean = max(list_mean_rule, key=lambda x: x['mean'])
    # print(f'best mean (%): {round(best_mean['mean'] * 100, 3)}')
    # print(f'best rule: {best_mean['rule']}, best_std: {best_mean['std']}')
    mean_max = list(filter(lambda x: x['rule'] == 'max', list_mean_rule))[0]['mean']
    std_max = list(filter(lambda x: x['rule'] == 'max', list_mean_rule))[0]['std']

    mean_prod = list(filter(lambda x: x['rule'] == 'prod', list_mean_rule))[0]['mean']
    std_prod = list(filter(lambda x: x['rule'] == 'prod', list_mean_rule))[0]['std']

    mean_sum = list(filter(lambda x: x['rule'] == 'sum', list_mean_rule))[0]['mean']
    std_sum = list(filter(lambda x: x['rule'] == 'sum', list_mean_rule))[0]['std']

    best_fold = max(list_result_fold, key=lambda x: x['accuracy'])
    # print(f'best acc (%): {round(best_fold['accuracy'] * 100, 3)}')
    # print(f'best fold: {best_fold['fold']}, best rule: {best_fold['rule']}')

    index = ['mean_time_sec', 'mean_time_millisec', 'mean_time_min', 'mean_time_hour_min_sec', 'std_time',
             'mean_sum', 'std_sum', 'mean_prod', 'std_prod', 'mean_max',
             'std_max', 'best_mean_rule', 'BEST_MEAN', 'best_mean_std', 'best_fold', 'best_rule', 'best_fold_accuracy',
             'best_params', 'n_features']
    values = [mean_time, mean_time_millisec, mean_time_min, mean_time_hour_min_sec, std_time, mean_sum, std_sum,
            mean_prod, std_prod, mean_max, std_max, best_mean['rule'], best_mean['mean'],
            best_mean['std'],
            best_fold['fold'], best_fold['rule'], best_fold['accuracy'], str(best_params), n_features]
    dataframe_mean = pandas.DataFrame(values, index)
    dataframe_mean.to_csv(os.path.join(path, 'mean.csv'), sep=';', na_rep='', header=False, quoting=csv.QUOTE_ALL)


def save_fold(cfg, classifier_name, dataset, list_result_fold, list_time, path):
    index = ['rule', 'accuracy']
    for f in range(0, cfg['fold']):
        list_fold = list(filter(lambda x: x['fold'] == f, list_result_fold))
        t = list(filter(lambda x: x['fold'] == f, list_time))

        list_rule = list()
        list_accuracy = list()
        path_fold = os.path.join(path, str(f))
        pathlib.Path(path_fold).mkdir(parents=True, exist_ok=True)

        for rule in list(['max', 'prod', 'sum']):
            result = list(filter(lambda x: x['rule'] == rule, list_fold))

            if len(result) > 0:
                r = result[0]
            else:
                r = list_fold[0]

            list_rule.append(rule)
            list_accuracy.append(r['accuracy'])

            save_confusion_matrix(classifier_name, dataset, path_fold, r)

        best_rule = max(list_fold, key=lambda x: x['accuracy'])

        dataframe_fold = pandas.DataFrame([list_rule, list_accuracy], index)
        dataframe_fold.to_csv(os.path.join(path_fold, 'out.csv'), sep=';', na_rep='', header=False, quoting=csv.QUOTE_ALL)

        time_sec = time.strftime('%H:%M:%S', time.gmtime(t[0]['time_train_valid']))
        dataframe_time = pandas.DataFrame([t[0]['time_train_valid'], time_sec], ['time', 'time_sec'])
        dataframe_best_rule = pandas.DataFrame([best_rule['rule'], best_rule['accuracy']],
                                               ['best_rule', 'best_accuracy'])
        dataframe_info = pandas.concat([dataframe_time, dataframe_best_rule])
        dataframe_info.to_csv(os.path.join(path_fold, 'fold_info.csv'), sep=';', na_rep='', header=False, quoting=csv.QUOTE_ALL)


def save_info_dataset(color_mode, data, dataset, dim, dir_input, extractor, n_patch, path, slice):
    values = [color_mode, data['x'].shape[1], data['x'].shape[0], dataset, dim, dir_input, extractor, n_patch, slice]
    index = ['olor_mode', 'data_n_features', 'data_n_samples', 'dataset', 'dim', 'dir_input', 'extractor', 'n_patch', 'slice']
    dataframe = pandas.DataFrame(values, index)
    dataframe.to_csv(os.path.join(path, 'info.csv'), sep=';', na_rep='', header=False, quoting=csv.QUOTE_ALL)


def save_confusion_matrix(classifier_name, dataset, path, result):
    filename = f'confusion_matrix-{result["rule"]}.png'
    labels = ['$\it{Manekia}$', '$\it{Ottonia}$', '$\it{Peperomia}$', '$\it{Piper}$', '$\it{Pothomorphe}$']
    confusion_matrix = sklearn.metrics.ConfusionMatrixDisplay(result['confusion_matrix'])
    confusion_matrix.plot(cmap='Reds')
    title = f'Confusion Matrix\ndataset: {dataset}, classifier: {classifier_name}\naccuracy: {round(result["accuracy"] * 100, 3)}, rule: {result["rule"]}'
    matplotlib.pyplot.ioff()
    matplotlib.pyplot.title(title, pad=20)
    matplotlib.pyplot.xticks(np.arange(5), labels, rotation=(45))
    matplotlib.pyplot.yticks(np.arange(5), labels)
    matplotlib.pyplot.ylabel('y_test', fontsize=12)
    matplotlib.pyplot.xlabel('y_pred', fontsize=12)
    matplotlib.pyplot.gcf().subplots_adjust(bottom=0.15, left=0.25)
    matplotlib.pyplot.rcParams['figure.facecolor'] = 'white'
    matplotlib.pyplot.rcParams['figure.figsize'] = (10, 10)
    matplotlib.pyplot.savefig(os.path.join(path, filename))
    matplotlib.pyplot.cla()
    matplotlib.pyplot.clf()
    matplotlib.pyplot.close()


if __name__ == '__main__':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['GOTO_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    main()


