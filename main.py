import click
import datetime
import joblib
import logging
import numpy as np
import os.path
import pathlib

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from dataset import load_dataset_informations, prepare_data, has_region
from fold import run_folds
from save import save_mean, mean_metrics, save_info, save_best, save_df_main

FOLDS = 5
GPU_ID = 0
METRIC = 'f1_weighted'
N_JOBS = -1
SEED = 1234
OUTPUT = '/home/xandao/results'
VERBOSE = 42

datefmt = '%d-%m-%Y+%H-%M-%S'
dateandtime = datetime.datetime.now().strftime(datefmt)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)

dimensions = {
    'mobilenetv2': [1280, 1024, 512, 256, 128],
    'vgg16': [512, 256, 128],
    'resnet50v2': [2048, 1024, 512, 256, 128],
    'lbp': [59],
    'surf64': [257, 256, 128],
    'surf128': [513, 512, 256, 128]
}

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
        'n_estimators': [200, 400, 600, 800, 1000],
        'max_features': ['sqrt', 'log2'],
        'criterion': ['gini', 'entropy'],
        'max_depth': [10, 100]
    },
    'SVC': {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }
}


def selected_classifier(classifiers_selected):
    classifiers = [
        DecisionTreeClassifier(random_state=SEED),
        KNeighborsClassifier(n_jobs=N_JOBS),
        MLPClassifier(random_state=SEED),
        RandomForestClassifier(random_state=SEED, n_jobs=N_JOBS, verbose=VERBOSE),
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
    classifiers_choosed = selected_classifier(classifiers)

    if len(classifiers_choosed) == 0:
        raise SystemExit('classifiers choosed not found')

    logging.info('[INFO] %s classifiers was choosed' % str(len(classifiers_choosed)))

    if not os.path.exists(input):
        raise SystemExit('input %s not found' % input)

    color, dataset, extractor, image_size, list_info_level, minimum_image, n_features, n_samples, patch = \
        load_dataset_informations(input)
    index, X, y = prepare_data(FOLDS, input, n_features, n_samples, patch, SEED)

    if np.isnan(X).any():
        raise SystemExit('X contains nan')

    if pca:
        list_x = [PCA(n_components=dim, random_state=SEED).fit_transform(X) for dim in dimensions[extractor.lower()] if
                  dim < n_features]
        list_x.append(X)
    else:
        list_x = [X]

    region = has_region(input)
    if region:
        logging.info('[INFO] exists a region %s' % region)

    logging.info('[INFO] result of pca %d' % len(list_x))
    list_results_classifiers = []
    for x in list_x:
        n_features = x.shape[1]

        for classifier in classifiers_choosed:
            results_fold = []
            classifier_name = classifier.__class__.__name__

            if region:
                output_folder_name = 'clf=%s+len=%s+ex=%s+ft=%s+c=%s+dt=%s+r=%s+m=%s' \
                                 % (classifier_name, str(image_size[0]), extractor, str(n_features), color, dataset,
                                    region, minimum_image)
            else:
                output_folder_name = 'clf=%s+len=%s+ex=%s+ft=%s+c=%s+dt=%s+m=%s' \
                                     % (classifier_name, str(image_size[0]), extractor, str(n_features), color, dataset,
                                        minimum_image)

            list_out_results = [str(p.name) for p in pathlib.Path(OUTPUT).rglob('*') if p.is_dir()]
            logging.info('encounter %s results' % len(list_out_results))

            if output_folder_name in list_out_results:
                logging.info('output_folder_name %s exists' % output_folder_name)
            else:
                path = os.path.join(OUTPUT, dateandtime, output_folder_name)

                if not os.path.exists(path):
                    os.makedirs(path)

                clf = GridSearchCV(classifier, parameters[classifier_name], cv=FOLDS,
                                   scoring=METRIC, n_jobs=N_JOBS, verbose=VERBOSE)

                with joblib.parallel_backend('loky', n_jobs=N_JOBS):
                    clf.fit(x, y)

                # enable to use predict_proba
                if isinstance(clf.best_estimator_, SVC):
                    params = dict(probability=True)
                    clf.best_estimator_.set_params(**params)

                for fold, i in enumerate(index, start=1):
                    index_train = i[0]
                    index_test = i[1]

                    params = {
                        'classifier': clf,
                        'classifier_name': classifier_name,
                        'fold': fold,
                        'index_train': index_train,
                        'index_test': index_test,
                        'list_info_level': list_info_level,
                        'n_features': n_features,
                        'patch': patch,
                        'path': path,
                        'x': x,
                        'y': y
                    }

                    result, n_labels = run_folds(**params)
                    results_fold.append(result)

                logging.info('results_fold %s' % str(len(results_fold)))
                means = mean_metrics(results_fold, n_labels)
                save_mean(means, path, results_fold)
                save_best(clf, means, path, results_fold)
                save_info(classifier_name, extractor, n_features, n_samples, path, patch)
                list_results_classifiers.append({
                    'classifier_name': classifier_name,
                    'image_size': str(image_size[0]),
                    'extractor': extractor,
                    'n_features': str(n_features),
                    'means': means
                })
                save_df_main(color, dataset, minimum_image, list_results_classifiers, OUTPUT, region=region)


if __name__ == '__main__':
    main()
