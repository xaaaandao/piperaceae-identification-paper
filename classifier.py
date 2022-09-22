import csv
import os
import numpy
import pandas
import pathlib

import sklearn.ensemble
import sklearn.exceptions
import sklearn.neighbors
import sklearn.neural_network
import sklearn.preprocessing
import sklearn.svm
import time
import warnings

from output import save
from result import calculate_test, convert_prob_to_label, max_all_results, \
    sum_all_results, prod_all_results, y_test_with_patch, y_pred_with_patch, create_result
from samples import get_samples_with_patch

warnings.simplefilter("ignore", category=sklearn.exceptions.ConvergenceWarning)

hyperparams = {
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
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    }
}


def data_has_patch(cfg, best_classifier, classifier_name, dataset, index, n_patch,
                   path_classifier, x, y):
    list_result_fold = list()
    list_time = list()
    for fold, (index_train, index_test) in enumerate(index):
        x_train, y_train = get_samples_with_patch(x, y, index_train, n_patch)
        x_test, y_test = get_samples_with_patch(x, y, index_test, n_patch)

        print(fold, classifier_name, x_train.shape, x_test.shape)

        start_time = time.time()
        best_classifier.fit(x_train, y_train)
        y_pred = best_classifier.predict_proba(x_test)
        end_time = time.time()

        result_max_rule, result_prod_rule, result_sum_rule = calculate_test(cfg, fold, y_pred, y_test, n_patch=n_patch)

        final_time = end_time - start_time

        list_result_fold.append(result_max_rule)
        list_result_fold.append(result_prod_rule)
        list_result_fold.append(result_sum_rule)
        list_time.append({
            "fold": fold,
            "final_time": final_time
        })

    return list_result_fold, list_time


def data_no_patch(cfg, best_classifier, classifier_name, dataset, index, path_classifier,
                  x, y):
    list_result_fold = list()
    list_time = list()
    for fold, (index_train, index_test) in enumerate(index):
        x_train, y_train = x[index_train], y[index_train]
        x_test, y_test = x[index_test], y[index_test]

        print(fold, classifier_name, x_train.shape, x_test.shape)

        start_time = time.time()
        best_classifier.fit(x_train, y_train)
        y_pred = best_classifier.predict_proba(x_test)
        end_time = time.time()

        result_max_rule, result_prod_rule, result_sum_rule = calculate_test(cfg, fold, y_pred, y_test)

        final_time = end_time - start_time

        list_result_fold.append(result_max_rule)
        list_result_fold.append(result_prod_rule)
        list_result_fold.append(result_sum_rule)
        list_time.append({
            "fold": fold,
            "final_time": final_time
        })

    return list_result_fold, list_time


def my_ensemble_classifier(cfg, dataset, list_result_classifier, n_features, n_samples, n_patch=None, orientation=None):
    classifier_name = "MyEnsembleClassifier"
    list_result_fold = list()
    list_time = list()

    if n_patch and orientation:
        path_classifier = os.path.join(cfg["path_out"], dataset, classifier_name, orientation, str(n_patch),
                                       str(n_features))
    else:
        path_classifier = os.path.join(cfg["path_out"], dataset, classifier_name, str(n_features))
    pathlib.Path(path_classifier).mkdir(parents=True, exist_ok=True)

    for fold in range(0, cfg["fold"]):
        # list_result_fold = get_result_per_attribute_and_value("fold", list_result_classifier, fold)
        list_fold = list(filter(lambda x: x["fold"] == fold, list_result_classifier))

        if len(list_fold) < 2:
            raise ValueError("number invalid of classifier")

        print(fold, classifier_name)

        y_true = list_fold[0]["y_true"]
        start_time = time.time()
        result_max_rule = create_result(fold, "max", None, max_all_results(list_fold), y_true)
        result_sum_rule = create_result(fold, "sum", None,
                                 convert_prob_to_label(sum_all_results(list_fold)), y_true)
        result_prod_rule = create_result(fold, "prod", None,
                                  convert_prob_to_label(prod_all_results(list_fold)), y_true)
        end_time = time.time()

        final_time = end_time - start_time

        list_result_fold.append(result_max_rule)
        list_result_fold.append(result_sum_rule)
        list_result_fold.append(result_prod_rule)
        list_time.append({
            "fold": fold,
            "final_time": final_time
        })

    save(None, cfg, classifier_name, dataset, list_result_fold, list_time, path_classifier)


def ensemble_classifier(cfg, dataset, index, list_best_classifiers, n_features, n_samples, path, x, y, n_patch=None,
                        orientation=None):
    classifier = sklearn.ensemble.VotingClassifier(estimators=list_best_classifiers, voting="hard")
    classifier_name = classifier.__class__.__name__

    list_result_fold = list()
    list_time = list()

    path = os.path.join(path, classifier_name, str(n_features))
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    for fold, (index_train, index_test) in enumerate(index):
        if n_patch and orientation:
            x_train, y_train = get_samples_with_patch(x, y, index_train, n_patch)
            x_test, y_test = get_samples_with_patch(x, y, index_test, n_patch)
        else:
            x_train, y_train = x[index_train], y[index_train]
            x_test, y_test = x[index_test], y[index_test]

        print(fold, classifier_name, x_train.shape, x_test.shape)

        start_time = time.time()
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        end_time = time.time()
        final_time = end_time - start_time

        if n_patch and orientation:
            y_true = y_test_with_patch(n_patch, y_test)
            y_pred = y_pred_with_patch(n_patch, y_pred)
        else:
            y_true = y_test

        r = create_result(fold, None, None, y_pred, y_true)
        list_time.append({
            "fold": fold,
            "final_time": final_time
        })
        list_result_fold.append(r)

    save(None, cfg, classifier_name, dataset, list_result_fold, list_time, path)

# classification_data(cfg, dataset, str(file), list(kf.split(x_surf)), pca, n_samples, path, x, x_surf, y, n_patch=n_patch, orientation=orientation)
def classification_data(cfg, dataset, file_input, index, n_features, n_samples, path, x, x_surf, y, n_patch=None, orientation=None):
    list_best_classifiers = list()
    list_result_classifier = list()

    for classifier in (
            sklearn.tree.DecisionTreeClassifier(random_state=cfg["random_state"]),
            sklearn.neighbors.KNeighborsClassifier(n_jobs=-1),
            sklearn.neural_network.MLPClassifier(random_state=cfg["random_state"]),
            sklearn.ensemble.RandomForestClassifier(random_state=cfg["random_state"], n_jobs=-1),
            sklearn.svm.SVC(random_state=cfg["random_state"], probability=True)):
        classifier_name = classifier.__class__.__name__
        print(numpy.unique(y))
        model = sklearn.model_selection.GridSearchCV(classifier, hyperparams[classifier_name], scoring="accuracy",
                                                     cv=index, n_jobs=-1, verbose=True)
        model.fit(x, y)

        best_classifier = model.best_estimator_
        best_params = model.best_params_

        list_best_classifiers.append((classifier_name, best_classifier))

        data = [file_input, n_features, n_samples, n_patch, orientation]
        columns = ["file_input", "n_features", "n_samples", "n_patch", "orientation"]
        dataframe = pandas.DataFrame(data, columns)
        dataframe.to_csv(os.path.join(path, "info.csv"), decimal=",", sep=";", na_rep=" ", header=False,
                          quoting=csv.QUOTE_ALL)

        path_completed = os.path.join(path, classifier_name, str(n_features))
        pathlib.Path(path_completed).mkdir(parents=True, exist_ok=True)

        if n_patch and orientation:
            list_result_fold, list_time = data_has_patch(cfg, best_classifier, classifier_name, dataset, list(index.split(x_surf)), n_patch,
                                              path_completed, x, y)
        else:
            list_result_fold, list_time = data_no_patch(cfg, best_classifier, classifier_name, dataset, list(index.split(x_surf)), path, x, y)

        save(best_params, cfg, classifier_name, dataset, list_result_fold, list_time, path_completed)
        list_result_classifier = list_result_classifier + list_result_fold

    # my_ensemble_classifier(cfg, dataset, list_result_classifier, n_features, n_samples, n_patch=n_patch, orientation=orientation)
    # ensemble_classifier(cfg, dataset, index, list_best_classifiers, n_features, n_samples, path, x, y, n_patch=n_patch,
    #                     orientation=orientation)
