import click
import collections
import datetime
import joblib
import logging
import multiprocessing
import numpy as np
import os.path
import ray
import timeit

from ray.util.joblib import register_ray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, top_k_accuracy_score, multilabel_confusion_matrix, \
    classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from arrays import mult_rule, split_dataset, sum_rule, y_true_no_patch, max_rule
from dataset import load_dataset_informations, prepare_data
from save import save_mean, save_fold, save_confusion_matrix, \
    mean_metrics, save_info, save_df_main, save_best

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


@ray.remote
class Fold:

    def __init__(self, classifier, classifier_name, fold, index_train, index_test, list_info_level, n_features, patch, path, results_fold, x, y):
        self.classifier = classifier
        self.classifier_name = classifier_name
        self.fold = fold
        self.list_info_level = list_info_level
        self.index_train = index_train
        self.index_test = index_test
        self.n_features = n_features
        self.patch = patch
        self.path = path
        self.results_fold = results_fold
        self.x = x
        self.y = y

    def run(self):
        logging.info('[INFO] fold: %d classifier name: %s' % (self.fold, self.classifier_name))
        x_train, y_train = split_dataset(self.index_train, self.n_features, self.patch, self.x, self.y)
        x_test, y_test = split_dataset(self.index_test, self.n_features, self.patch, self.x, self.y)

        count_train = sorted(collections.Counter(y_train).items())
        count_test = sorted(collections.Counter(y_test).items())
        logging.info('[INFO] TRAIN: %s ' % str(count_train))
        logging.info('[INFO] TEST: %s ' % str(count_test))

        logging.info('[INFO] x_train.shape: %s y_train.shape: %s' % (str(x_train.shape), str(y_train.shape)))
        logging.info('[INFO] x_test.shape: %s y_test.shape: %s' % (str(x_test.shape), str(y_test.shape)))

        start_timeit = timeit.default_timer()
        # with joblib.parallel_backend('ray', n_jobs=N_JOBS):
        self.classifier.best_estimator_.fit(x_train, y_train)

        y_pred_proba = self.classifier.best_estimator_.predict_proba(x_test)
        end_timeit = timeit.default_timer() - start_timeit

        n_test, n_labels = y_pred_proba.shape
        y_pred_max_rule, y_score_max = max_rule(n_test, n_labels, self.patch, y_pred_proba)
        y_pred_mult_rule, y_score_mult = mult_rule(n_test, n_labels, self.patch, y_pred_proba)
        y_pred_sum_rule, y_score_sum = sum_rule(n_test, n_labels, self.patch, y_pred_proba)
        y_true = y_true_no_patch(n_test, self.patch, y_test)

        logging.info('[INFO] y_pred_sum_rule.shape: %s y_score_sum: %s' % (
            str(y_pred_sum_rule.shape), str(y_score_sum.shape)))
        logging.info('[INFO] y_pred_mult_rule.shape: %s y_score_mult: %s' % (
            str(y_pred_mult_rule.shape), str(y_score_mult.shape)))
        logging.info('[INFO] y_pred_max_rule.shape: %s y_score_max: %s' % (
            str(y_pred_max_rule.shape), str(y_score_max.shape)))
        logging.info('[INFO] y_true.shape: %s' % str(y_true.shape))

        results = {
            'fold': self.fold,
            'max': evaluate(self.list_info_level, n_labels, y_pred_max_rule, y_score_max, y_true),
            'mult': evaluate(self.list_info_level, n_labels, y_pred_mult_rule, y_score_mult, y_true),
            'sum': evaluate(self.list_info_level, n_labels, y_pred_sum_rule, y_score_sum, y_true),
            'time': end_timeit
        }

        logging.info('results_fold %s' % str(len(self.results_fold)))

        path_fold = os.path.join(self.path, str(self.fold))

        if not os.path.exists(path_fold):
            os.makedirs(path_fold)

        save_fold(count_train, count_test, self.fold, path_fold, results)
        save_confusion_matrix(self.list_info_level, path_fold, results)
        return {
            'results': results,
            'n_labels': n_labels
        }


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

    if input.endswith('.txt') and os.path.isfile(input):
        pass
    else:
        color, dataset, extractor, image_size, list_info_level, minimum_image, n_features, n_samples, patch = load_dataset_informations(input)
        index, x, y = prepare_data(FOLDS, input, n_features, n_samples, patch, SEED)
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

                #     # enable to use predict_proba
                #     if isinstance(clf.best_estimator_, SVC):
                #         params = dict(probability=True)
                #         clf.best_estimator_.set_params(**params)
                #
                folds = []
                for fold, (index_train, index_test) in enumerate(index, start=1):
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
                    # folds = [ ]

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
                    # print(len(finished))
                    #
                    # for t in finished:
                    #     result = ray.get(t)
                    #     print(type(result))


                # logging.info('[INFO] length of list_results %d' % len(list_results))
                # if len(list_results) == 0:
                #     raise SystemExit('error ray.wait and ray.get')
                #
                # results_fold = [r['results'] for r in list_results]
                # n_labels = list_results[0]['n_labels']
                # means = mean_metrics(results_fold, n_labels)
                # save_mean(means, path, results_fold)
                # save_best(clf, means, path, results_fold)
                # save_info(classifier.__class__.__name__, extractor, n_features, n_samples, path, patch)
                # list_results_classifiers.append({
                #     'classifier_name': classifier.__class__.__name__,
                #     'image_size': str(image_size[0]),
                #     'extractor': extractor,
                #     'n_features': str(n_features),
                #     'means': means
                # })
            #         logging.info('[INFO] fold: %d classifier name: %s' % (fold, classifier_name))
            #         x_train, y_train = split_dataset(index_train, n_features, patch, x, y)
            #         x_test, y_test = split_dataset(index_test, n_features, patch, x, y)
            #
            #         count_train = sorted(collections.Counter(y_train).items())
            #         count_test = sorted(collections.Counter(y_test).items())
            #         logging.info('[INFO] TRAIN: %s ' % str(count_train))
            #         logging.info('[INFO] TEST: %s ' % str(count_test))
            #
            #         logging.info('[INFO] x_train.shape: %s y_train.shape: %s' % (str(x_train.shape), str(y_train.shape)))
            #         logging.info('[INFO] x_test.shape: %s y_test.shape: %s' % (str(x_test.shape), str(y_test.shape)))
            #
            #         start_timeit = timeit.default_timer()
            #         with joblib.parallel_backend('ray', n_jobs=N_JOBS):
            #             clf.best_estimator_.fit(x_train, y_train)
            #
            #         y_pred_proba = clf.best_estimator_.predict_proba(x_test)
            #         end_timeit = timeit.default_timer() - start_timeit
            #
            #         n_test, n_labels = y_pred_proba.shape
            #         y_pred_max_rule, y_score_max = max_rule(n_test, n_labels, patch, y_pred_proba)
            #         y_pred_mult_rule, y_score_mult = mult_rule(n_test, n_labels, patch, y_pred_proba)
            #         y_pred_sum_rule, y_score_sum = sum_rule(n_test, n_labels, patch, y_pred_proba)
            #         y_true = y_true_no_patch(n_test, patch, y_test)
            #
            #         logging.info('[INFO] y_pred_sum_rule.shape: %s y_score_sum: %s' % (
            #             str(y_pred_sum_rule.shape), str(y_score_sum.shape)))
            #         logging.info('[INFO] y_pred_mult_rule.shape: %s y_score_mult: %s' % (
            #             str(y_pred_mult_rule.shape), str(y_score_mult.shape)))
            #         logging.info('[INFO] y_pred_max_rule.shape: %s y_score_max: %s' % (
            #             str(y_pred_max_rule.shape), str(y_score_max.shape)))
            #         logging.info('[INFO] y_true.shape: %s' % str(y_true.shape))
            #
            #         results = {
            #             'fold': fold,
            #             'max': evaluate(list_info_level, n_labels, y_pred_max_rule, y_score_max, y_true),
            #             'mult': evaluate(list_info_level, n_labels, y_pred_mult_rule, y_score_mult, y_true),
            #             'sum': evaluate(list_info_level, n_labels, y_pred_sum_rule, y_score_sum, y_true),
            #             'time': end_timeit
            #         }
            #
            #         results_fold.append(results)
            #         logging.info('results_fold %s' % str(len(results_fold)))
            #
            #         path_fold = os.path.join(path, str(fold))
            #
            #         if not os.path.exists(path_fold):
            #             os.makedirs(path_fold)
            #
            #         save_fold(count_train, count_test, fold, path_fold, results)
            #         save_confusion_matrix(list_info_level, path_fold, results)
            #     means = mean_metrics(results_fold, n_labels)
            #     save_mean(means, path, results_fold)
            #     save_best(clf, means, path, results_fold)
            #     save_info(classifier.__class__.__name__, extractor, n_features, n_samples, path, patch)
            #     list_results_classifiers.append({
            #         'classifier_name': classifier.__class__.__name__,
            #         'image_size': str(image_size[0]),
            #         'extractor': extractor,
            #         'n_features': str(n_features),
            #         'means': means
            #     })
            save_df_main(dataset, dimensions, minimum_image, list_results_classifiers, OUTPUT)


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


if __name__ == '__main__':
    main()
