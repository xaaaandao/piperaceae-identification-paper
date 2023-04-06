import collections
import joblib
import itertools
import multiprocessing
import numpy as np
import os.path
import pandas as pd
import pathlib
import ray

from ray.util.joblib import register_ray
from sklearn.metrics import confusion_matrix, f1_score, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from a import mult_rule, split_dataset, sum_rule, y_true_no_patch
from save import save_mean, save_best_mean, save_fold, save_info, save_best_classifier, save_confusion_matrix

FOLDS = 5
METRIC = 'f1_weighted'
N_JOBS = -1
PCA = False
SEED = 1234
OUTPUT = '/home/xandao/Documentos/resultados'

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

list_classifiers = [
    # KNeighborsClassifier(n_jobs=N_JOBS),
    # MLPClassifier(random_state=SEED),
    # RandomForestClassifier(random_state=SEED, n_jobs=N_JOBS, verbose=100, max_depth=10),
    # SVC(random_state=SEED, verbose=True, cache_size=2000, C=0.01)
    DecisionTreeClassifier(random_state=SEED),
]


def main():
    # input='lbp.txt'
    input = '/home/xandao/Imagens/pr_dataset_features/RGB/256/specific_epithet_trusted/20/vgg16'
    if input.endswith('.txt') and os.path.isfile(input):
        data = np.loadtxt(input)
        n_samples, n_features = data.shape
        x, y = data[0:, 0:n_features - 1], data[:, n_features - 1]

        if np.isnan(x):
            raise SystemExit('dataset %s contains nan' % input)

        kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
        index = kf.split(x, y)
        for fold, (index_train, index_test) in enumerate(index):
            print('fold: %d' % fold)
            x_train, y_train = x[index_train], y[index_train]
            x_test, y_test = x[index_test], y[index_test]

    else:
        extractor, list_info_level, n_features, n_samples, patch = read_dataset_informations(input)
        index, x, y = prepare_data(input, n_features, n_samples, patch)
        # print(np.unique(y))
        for classifier in list_classifiers:
            list_results = []
            output_folder_name = '%s+%s+%s' % (classifier.__class__.__name__, extractor, str(n_features))
            path = os.path.join(OUTPUT, output_folder_name)

            if not os.path.exists(path):
                os.makedirs(path)

            clf = GridSearchCV(classifier, parameters[classifier.__class__.__name__], cv=FOLDS,
                               scoring=METRIC, n_jobs=N_JOBS, verbose=True)

            with joblib.parallel_backend('ray', n_jobs=N_JOBS):
                clf.fit(x, y)

            for fold, (index_train, index_test) in enumerate(index):
                x_train, y_train = split_dataset(index_train, n_features, patch, x, y)
                x_test, y_test = split_dataset(index_test, n_features, patch, x, y)

                # print(collections.Counter(y_test))
                # print(np.unique(y_test))
                print('fold %d classifier name: %s' % (fold, classifier.__class__.__name__))
                print('x_train.shape: %s y_train.shape: %s' % (str(x_train.shape), str(y_train.shape)))
                print('x_test.shape: %s y_test.shape: %s' % (str(x_test.shape), str(y_test.shape)))

                clf.best_estimator_.fit(x_train, y_train)
                y_pred_proba = clf.best_estimator_.predict_proba(x_test)
                n_test, n_labels = y_pred_proba.shape

                y_pred_mult_rule = mult_rule(n_test, n_labels, patch, y_pred_proba)
                y_pred_sum_rule = sum_rule(n_test, n_labels, patch, y_pred_proba)
                # print(y_test)
                y_true = y_true_no_patch(n_test, patch, y_test)
                # print(y_true)
                print('y_pred_sum_rule.shape: %s' % str(y_pred_sum_rule.shape))
                print('y_pred_mult_rule.shape %s' % str(y_pred_mult_rule.shape))
                print('y_true.shape %s' % str(y_true.shape))

                #
                results = {
                    'mult': evaluate(y_pred_mult_rule, y_true),
                    'sum': evaluate(y_pred_sum_rule, y_true)
                }
                #
                # list_results.append(results)
                # path_fold = os.path.join(path, str(fold))
                #
                # if not os.path.exists(path_fold):
                #     os.makedirs(path_fold)

                # save_fold(fold, path_fold, results)
                # save_best_classifier(clf, path_fold)
                # save_confusion_matrix(list_info_level, path_fold, results)
                break
            break
            # save_mean(list_results, path)
            # save_best_fold(list_results, path)
            # save_best_mean(list_results, path)
            # save_info(classifier.__class__.__name__, extractor, n_features, n_samples, path, patch)


def evaluate(y_pred, y_true):
    f1 = f1_score(y_true, y_pred, average='weighted')
    topk_three = None #top_k_accuracy_score(y_true, k=3, normalize=True)
    topk_five = None #top_k_accuracy_score(y_true, k=5, normalize=True)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    return {'f1': f1, 'topk_three': topk_three, 'topk_five': topk_five, 'confusion_matrix': cm}


def read_dataset_informations(input):
    info_dataset = [f for f in pathlib.Path(input).rglob('info.csv') if f.is_file()]

    if len(info_dataset) == 0:
        raise SystemExit('info.csv not found in %s' % input)

    df = pd.read_csv(info_dataset[0], index_col=0, header=None, sep=';')
    extractor = df.loc['cnn'][1]
    input_path = df.loc['input_path'][1]
    n_features = int(df.loc['n_features'][1])
    n_samples = int(df.loc['total_samples'][1])
    patch = int(df.loc['patch'][1])
    print('n_samples: %s n_features: %s patch: %s' % (n_samples, n_features, patch))

    input_path = input_path.replace('/media/kingston500/mestrado/dataset', '/home/xandao/Imagens/')
    if not os.path.exists(input_path):
        raise SystemExit('input path %s not exists' % input_path)

    info_level = [f for f in pathlib.Path(input_path).rglob('info_levels.csv') if f.is_file()]

    if len(info_dataset) == 0:
        raise SystemExit('info_levels.csv not found in %s' % input)

    df = pd.read_csv(info_level[0], header=0, sep=';')
    list_info_level = df[['levels', 'count', 'f']].to_dict()

    return extractor, list_info_level, n_features, n_samples, patch


def prepare_data(input, n_features, n_samples, patch):
    x = np.empty(shape=(n_samples, n_features), dtype=np.float64)
    y = []

    for file in sorted(pathlib.Path(input).rglob('*.npz')):
        if file.is_file():
            d = np.load(file)
            np.vstack((x, d['x']))
            y.append(d['y'])

    y = np.array(list(itertools.chain(*y)), dtype=np.int16)
    print('dataset contains x.shape: %s' % str(x.shape))
    print('dataset contains y.shape: %s' % str(y.shape))

    index = split_folds(n_features, n_samples, patch, y)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return index, x, y


def split_folds(n_features, n_samples, patch, y):
    np.random.seed(SEED)
    x = np.random.rand(int(n_samples/patch), n_features)
    y = [np.repeat(k, int(v/patch)) for k, v in dict(collections.Counter(y)).items()]
    y = np.array(list(itertools.chain(*y)))
    print('StratifiedKFold x.shape: %s' % str(x.shape))
    print('StratifiedKFold y.shape: %s' % str(y.shape))
    kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    return kf.split(x, y)


if __name__ == '__main__':
    main()
