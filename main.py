import click
import datetime
import joblib
import logging
import multiprocessing
import os.path
import ray

from ray.util.joblib import register_ray
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from arrays import load_data_txt
from dataset import load_dataset_informations, prepare_data
from fold import Fold
from save import save_mean, mean_metrics, save_info, save_df_main, save_best

FOLDS = 5
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
        'n_estimators': [250, 500, 750, 1000],
        'max_features': ['sqrt', 'log2'],
        'criterion': ['gini', 'entropy']
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
    classifiers_choosed = selected_classifier(classifiers)

    if len(classifiers_choosed) == 0:
        raise SystemExit('classifiers choosed not found')

    logging.info('[INFO] %s classifiers was choosed' % str(len(classifiers_choosed)))

    if not os.path.exists(input):
        raise SystemExit('input %s not found' % input)

    color, dataset, extractor, image_size, list_info_level, minimum_image, n_features, n_samples, patch = load_dataset_informations(input)
    index, x, y = prepare_data(FOLDS, input, n_features, n_samples, patch, SEED)

    if pca:
        list_x = [PCA(n_components=dim, random_state=SEED).fit_transform(x) for dim in dimensions[extractor.lower()] if dim < n_features]
        list_x.append(x)
    else:
        list_x = [x]

    logging.info('[INFO] result of pca %d' % len(list_x))
    list_results_classifiers = []
    for x in list_x:
        n_features = x.shape[1]
        for classifier in classifiers_choosed:
            results_fold = []
            classifier_name = classifier.__class__.__name__

            clf = GridSearchCV(classifier, parameters[classifier_name], cv=FOLDS,
                               scoring=METRIC, n_jobs=N_JOBS, verbose=VERBOSE)

            with joblib.parallel_backend('ray', n_jobs=N_JOBS):
                clf.fit(x, y)

            # enable to use predict_proba
            if isinstance(clf.best_estimator_, SVC):
                params = dict(probability=True)
                clf.best_estimator_.set_params(**params)

            folds = []
            for fold, (index_train, index_test) in enumerate(index, start=1):
                output_folder_name = 'clf=%s+len=%s+ex=%s+ft=%s+c=%s+dt=%s+m=%s' \
                                     % (classifier_name, str(image_size[0]), extractor, str(n_features), color, dataset,
                                        minimum_image)
                path = os.path.join(OUTPUT, dateandtime, output_folder_name)

                if not os.path.exists(path):
                    os.makedirs(path)

                params = {
                    'classifier': clf,
                    'classifier_name': classifier_name,
                    'fold': fold,
                    'list_info_level': list_info_level,
                    'index_train': index_train,
                    'index_test': index_test,
                    'n_features': n_features,
                    'patch': patch,
                    'path': path,
                    'results_fold': results_fold,
                    'x': x,
                    'y': y
                }
                folds.append(Fold.remote(**params))

            run_folds = [f.run.remote() for f in folds]
            while run_folds:
                finished, run_folds = ray.wait(run_folds, num_returns=len(run_folds))

                if len(finished) == 0:
                    raise SystemExit('error ray.wait and ray.get')

                results_fold = [ray.get(t)['results'] for t in finished]
                n_labels = ray.get(finished[0])['n_labels']
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
        save_df_main(dataset, dimensions, minimum_image, list_results_classifiers, OUTPUT)



if __name__ == '__main__':
    main()
