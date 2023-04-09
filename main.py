import click
import collections
import datetime
import joblib
import itertools
import logging
import multiprocessing
import numpy as np
import os.path
import pandas as pd
import pathlib
import ray
import timeit

from ray.util.joblib import register_ray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, top_k_accuracy_score, multilabel_confusion_matrix, \
    classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from arrays import mult_rule, split_dataset, sum_rule, y_true_no_patch, max_rule
from save import save_mean, save_fold, save_confusion_matrix, \
    mean_metrics, save_info, save_df_main, save_best

FOLDS = 2
METRIC = 'f1_weighted'
N_JOBS = -1
SEED = 1234
OUTPUT = '/home/xandao/results'
VERBOSE = 42

datefmt = '%d-%m-%Y+%H-%M-%S'
dateandtime = datetime.datetime.now().strftime(datefmt)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)

ray.init(num_gpus=1, num_cpus=int(multiprocessing.cpu_count() / 2))
register_ray()

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
        'n_estimators': [250, 500, 750, 1000],
        'max_features': ['sqrt', 'log2'],
        'criterion': ['gini', 'entropy']
    },
    'SVC': {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }
}


dimensions = {
    'mobilenetv2': [1280, 1024, 512, 256, 128],
    'vgg16': [512, 256, 128],
    'resnet50v2': [2048, 1024, 512, 256, 128],
    'lbp': [59],
    'surf64': [257, 256, 128],
    'surf128': [513, 512, 256, 128]
}


def selected_classifier(classifiers_selected):
    classifiers = [
        DecisionTreeClassifier(random_state=SEED),
        KNeighborsClassifier(n_jobs=N_JOBS),
        MLPClassifier(random_state=SEED),
        RandomForestClassifier(random_state=SEED, n_jobs=N_JOBS, verbose=VERBOSE, max_depth=10),
        SVC(random_state=SEED, verbose=VERBOSE, cache_size=2000, C=0.01)
    ]
    return [c for cs in classifiers_selected for c in classifiers if cs == c.__class__.__name__]


@click.command()
@click.option('-c', '--classifiers', multiple=True, type=click.Choice(
    ['DecisionTreeClassifier', 'RandomForestClassifier', 'KNeighborsClassifier', 'MLPClassifier', 'SVC']),
              default=['DecisionTreeClassifier'])
@click.option('-i', '--input',
              default='/home/xandao/Imagens/pr_dataset_features/RGB/256/specific_epithet_trusted/20/vgg16')
@click.option('-p', '--pca', is_flag=True, default=False)
def main(classifiers, input, pca):
    print(pca)
    classifiers_choosed = selected_classifier(classifiers)

    if len(classifiers_choosed) == 0:
        raise SystemExit('classifiers choosed not found')

    logging.info('[INFO] %s classifiers was choosed' % str(len(classifiers_choosed)))

    if not os.path.exists(input):
        raise SystemExit('input %s not found' % input)

    if input.endswith('.txt') and os.path.isfile(input):
        pass
    else:
        color, dataset, extractor, image_size, list_info_level, minimum_image, n_features, n_samples, patch = load_dataset_informations(input)
        index, x, y = prepare_data(input, n_features, n_samples, patch)
        list_results_classifiers = []

        if pca:
            list_x = [PCA(n_components=dim, random_state=SEED).fit_transform(x) for dim in dimensions[extractor]]
            list_x.append(x)
        else:
            list_x = [x]

        for x in list_x:
            n_features = x.shape[1]
            for classifier in classifiers_choosed:
                results_fold = []
                classifier_name = classifier.__class__.__name__
                output_folder_name = 'clf=%s+size=%s+ex=%s+ft=%s+c=%s+dt=%s' % (classifier_name, str(image_size[0]), extractor, str(n_features), color, dataset)
                path = os.path.join(OUTPUT, dateandtime, output_folder_name)

                if not os.path.exists(path):
                    os.makedirs(path)

                clf = GridSearchCV(classifier, parameters[classifier_name], cv=FOLDS,
                                   scoring=METRIC, n_jobs=N_JOBS, verbose=VERBOSE)

                with joblib.parallel_backend('ray', n_jobs=N_JOBS):
                    clf.fit(x, y)

                # enable to use predict_proba
                if isinstance(clf.best_estimator_, SVC):
                    params = dict(probability=True)
                    clf.best_estimator_.set_params(**params)

                for fold, (index_train, index_test) in enumerate(index, start=1):
                    logging.info('[INFO] fold: %d classifier name: %s' % (fold, classifier_name))
                    x_train, y_train = split_dataset(index_train, n_features, patch, x, y)
                    x_test, y_test = split_dataset(index_test, n_features, patch, x, y)

                    count_train = sorted(collections.Counter(y_train).items())
                    count_test = sorted(collections.Counter(y_test).items())
                    logging.info('[INFO] TRAIN: %s ' % str(count_train))
                    logging.info('[INFO] TEST: %s ' % str(count_test))

                    logging.info('[INFO] x_train.shape: %s y_train.shape: %s' % (str(x_train.shape), str(y_train.shape)))
                    logging.info('[INFO] x_test.shape: %s y_test.shape: %s' % (str(x_test.shape), str(y_test.shape)))

                    start_timeit = timeit.default_timer()
                    with joblib.parallel_backend('ray', n_jobs=N_JOBS):
                        clf.best_estimator_.fit(x_train, y_train)

                    y_pred_proba = clf.best_estimator_.predict_proba(x_test)
                    end_timeit = timeit.default_timer() - start_timeit

                    n_test, n_labels = y_pred_proba.shape
                    y_pred_max_rule, y_score_max = max_rule(n_test, n_labels, patch, y_pred_proba)
                    y_pred_mult_rule, y_score_mult = mult_rule(n_test, n_labels, patch, y_pred_proba)
                    y_pred_sum_rule, y_score_sum = sum_rule(n_test, n_labels, patch, y_pred_proba)
                    y_true = y_true_no_patch(n_test, patch, y_test)

                    logging.info('[INFO] y_pred_sum_rule.shape: %s y_score_sum: %s' % (
                        str(y_pred_sum_rule.shape), str(y_score_sum.shape)))
                    logging.info('[INFO] y_pred_mult_rule.shape: %s y_score_mult: %s' % (
                        str(y_pred_mult_rule.shape), str(y_score_mult.shape)))
                    logging.info('[INFO] y_pred_max_rule.shape: %s y_score_max: %s' % (
                        str(y_pred_max_rule.shape), str(y_score_max.shape)))
                    logging.info('[INFO] y_true.shape: %s' % str(y_true.shape))

                    results = {
                        'fold': fold,
                        'max': evaluate(list_info_level, n_labels, y_pred_max_rule, y_score_max, y_true),
                        'mult': evaluate(list_info_level, n_labels, y_pred_mult_rule, y_score_mult, y_true),
                        'sum': evaluate(list_info_level, n_labels, y_pred_sum_rule, y_score_sum, y_true),
                        'time': end_timeit
                    }

                    results_fold.append(results)
                    logging.info('results_fold %s' % str(len(results_fold)))
                    path_fold = os.path.join(path, str(fold))

                    if not os.path.exists(path_fold):
                        os.makedirs(path_fold)

                    save_fold(count_train, count_test, fold, path_fold, results)
                    save_confusion_matrix(list_info_level, path_fold, results)
                means = mean_metrics(results_fold, n_labels)
                save_mean(means, path, results_fold)
                save_best(clf, means, path, results_fold)
                save_info(classifier.__class__.__name__, extractor, n_features, n_samples, path, patch)
                list_results_classifiers.append({
                    'classifier_name': classifier.__class__.__name__,
                    'image_size': str(image_size[0]),
                    'extractor': extractor,
                    'n_features': str(n_features),
                    'means': means
                })
            save_df_main(dimensions, minimum_image, list_results_classifiers, OUTPUT)


def evaluate(list_info_level, n_labels, y_pred, y_score, y_true):
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    accuracy = accuracy_score(y_pred=y_pred, y_true=y_true)
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    cm_normalized = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true')
    cm_multilabel = multilabel_confusion_matrix(y_pred=y_pred, y_true=y_true)
    cr = classification_report(y_pred=y_pred, y_true=y_true, labels=np.arange(1, len(list_info_level['levels']) + 1),
                               target_names=['label_%s+%s' % (i, label) for i, label in
                                             enumerate(list_info_level['levels'].values(), start=1)], zero_division=0,
                               output_dict=True)
    list_topk = [
        {'k': k,
         'top_k_accuracy': top_k_accuracy_score(y_true=y_true, y_score=y_score, normalize=False, k=k,
                                                labels=np.arange(1, len(list_info_level['levels']) + 1))}
        for k in range(3, n_labels)
    ]
    return {'f1': f1,
            'confusion_matrix': cm,
            'confusion_matrix_normalized': cm_normalized,
            'confusion_matrix_multilabel': cm_multilabel,
            'classification_report': cr, 'list_topk': list_topk, 'accuracy': accuracy}


def load_dataset_informations(input):
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

    input_path = input_path.replace('_features', '')
    # input_path = input_path.replace('/media/kingston500/mestrado/dataset', '/home/xandao/Imagens')
    if not os.path.exists(input_path):
        raise SystemExit('input path %s not exists' % input_path)

    info_level = [f for f in pathlib.Path(input_path).rglob('info_levels.csv') if f.is_file()]

    if len(info_dataset) == 0:
        raise SystemExit('info_levels.csv not found in %s' % input)

    logging.info('[INFO] reading file %s' % str(info_level[0]))
    df = pd.read_csv(info_level[0], index_col=None, usecols=['levels', 'count', 'f'], sep=';')
    list_info_level = df[['levels', 'count', 'f']].to_dict()

    logging.info('[INFO] n_levels: %s' % str(len(list_info_level['levels'])))

    return color, dataset, extractor, (height, width), list_info_level, minimum_image, n_features, n_samples, patch


def prepare_data(input, n_features, n_samples, patch):
    x = np.empty(shape=(0, n_features), dtype=np.float64)
    y = []

    for file in sorted(pathlib.Path(input).rglob('*.npz')):
        if file.is_file():
            d = np.load(file)
            x = np.append(x, d['x'], axis=0)
            y.append(d['y'])

    y = np.array(list(itertools.chain(*y)), dtype=np.int16)
    logging.info('[INFO] dataset contains x.shape: %s' % str(x.shape))
    logging.info('[INFO] dataset contains y.shape: %s' % str(y.shape))

    index = split_folds(n_features, n_samples, patch, y)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return index, x, y


def split_folds(n_features, n_samples, patch, y):
    np.random.seed(SEED)
    x = np.random.rand(int(n_samples / patch), n_features)
    y = [np.repeat(k, int(v / patch)) for k, v in dict(collections.Counter(y)).items()]
    y = np.array(list(itertools.chain(*y)))
    logging.info('[INFO] StratifiedKFold x.shape: %s' % str(x.shape))
    logging.info('[INFO] StratifiedKFold y.shape: %s' % str(y.shape))
    kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    return kf.split(x, y)


if __name__ == '__main__':
    main()
