import csv
import datetime
import os
import pandas
import pathlib

import numpy
import sklearn.ensemble
import sklearn.exceptions
import sklearn.neighbors
import sklearn.neural_network
import sklearn.preprocessing
import sklearn.svm
import time
import warnings

from output import save_fold, save_mean_std, save_confusion_matrix
from result import calculate_test, get_result_per_attribute_and_value, Result, convert_prob_to_label, max_all_results, \
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

        # path_fold = os.path.join(path_classifier, str(fold))
        # pathlib.Path(path_fold).mkdir(parents=True, exist_ok=True)

        result_max_rule, result_prod_rule, result_sum_rule = calculate_test(cfg, fold, y_pred, y_test, n_patch=n_patch)

        final_time = end_time - start_time

        list_result_fold.append(result_max_rule)
        list_result_fold.append(result_prod_rule)
        list_result_fold.append(result_sum_rule)
        list_time.append({
            "fold": fold,
            "final_time": final_time
        })

        # save_fold(classifier_name, dataset, final_time, (result_max_rule, result_prod_rule, result_sum_rule),
        #           path_fold)
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

        # path_fold = os.path.join(path_classifier, str(fold))
        # pathlib.Path(path_fold).mkdir(parents=True, exist_ok=True)

        result_max_rule, result_prod_rule, result_sum_rule = calculate_test(cfg, fold, y_pred, y_test)

        final_time = end_time - start_time

        list_result_fold.append(result_max_rule)
        list_result_fold.append(result_prod_rule)
        list_result_fold.append(result_sum_rule)
        list_time.append({
            "fold": fold,
            "final_time": final_time
        })

        # save_fold(classifier_name, dataset, final_time, (result_max_rule, result_prod_rule, result_sum_rule), path_fold)
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
        # result_max_rule = Result(fold, "max", None, max_all_results(list_fold), y_true)
        # result_sum_rule = Result(fold, "sum", None,
        #                          convert_prob_to_label(sum_all_results(list_fold)), y_true)
        # result_prod_rule = Result(fold, "prod", None,
        #                           convert_prob_to_label(prod_all_results(list_fold)), y_true)
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
    print("\n")


def ensemble_classifier(cfg, dataset, index, list_best_classifiers, n_features, n_samples, path, x, y, n_patch=None,
                        orientation=None):
    classifier = sklearn.ensemble.VotingClassifier(estimators=list_best_classifiers, voting="hard")
    classifier_name = classifier.__class__.__name__

    list_result_fold = list()
    list_time = list()

    if n_patch and orientation:
        path_classifier = os.path.join(cfg["path_out"], dataset, classifier_name, orientation, str(n_patch),
                                       str(n_features))
    else:
        path_classifier = os.path.join(cfg["path_out"], dataset, classifier_name, str(n_features))
    pathlib.Path(path_classifier).mkdir(parents=True, exist_ok=True)

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

        # r = Result(fold, None, None, y_pred, y_true)
        r = create_result(fold, None, None, y_pred, y_true)
        list_time.append({
            "fold": fold,
            "final_time": final_time
        })
        list_result_fold.append(r)

    save(None, cfg, classifier_name, dataset, list_result_fold, list_time, path)
    print("\n")
    #     path_fold = os.path.join(path_classifier, str(fold))
    #     pathlib.Path(path_fold).mkdir(parents=True, exist_ok=True)
    #     save_fold(classifier_name, dataset, final_time, list([r]), path_fold)
    # save_mean_std(None, cfg, list_result_fold, list_time, n_features, n_samples, path)


def save(best_params, cfg, classifier_name, dataset, list_result_fold, list_time, path):
    save2(cfg, classifier_name, dataset, list_result_fold, list_time, path)
    save_mean(best_params, list_result_fold, list_time, path)


def save_mean(best_params, list_result_fold, list_time, path):
    best_fold = max(list_result_fold, key=lambda x: x["accuracy"])
    mean_time = numpy.mean([t["final_time"] for t in list_time])
    mean_time_sec = time.strftime("%H:%M:%S", time.gmtime(mean_time))
    std_time = numpy.std([t["final_time"] for t in list_time])
    dataframe_mean = pandas.DataFrame(
        [mean_time, mean_time_sec, std_time, best_fold["fold"], best_fold["accuracy"], str(best_params)],
        ["mean_time", "mean_time_sec", "std_time", "best_fold", "best_fold_accuracy", "best_params"])
    dataframe_mean.to_csv(os.path.join(path, "mean.csv"), decimal=",", sep=";", na_rep=" ", header=False,
                          quoting=csv.QUOTE_ALL)


def save2(cfg, classifier_name, dataset, list_result_fold, list_time, path):
    columns = ["rule", "accuracy", "accuracy_per"]
    for f in range(0, cfg["fold"]):
        list_fold = list(filter(lambda x: x["fold"] == f, list_result_fold))
        t = list(filter(lambda x: x["fold"] == f, list_time))

        list_rule = list()
        list_accuracy = list()
        list_accuracy_per = list()
        path_fold = os.path.join(path, str(f))
        pathlib.Path(path_fold).mkdir(parents=True, exist_ok=True)
        for rule in list(["max", "prod", "sum"]):
            result = list(filter(lambda x: x["rule"] == rule, list_fold))

            if len(result) > 0:
                r = result[0]
            else:
                r = list_fold[0]

            list_rule.append(rule)
            list_accuracy.append(r["accuracy"])
            list_accuracy_per.append(round(r["accuracy"] * 100, 4))
            save_confusion_matrix(classifier_name, dataset, path_fold, r)

        best_rule = max(list_fold, key=lambda x: x["accuracy"])

        dataframe_fold = pandas.DataFrame([list_rule, list_accuracy, list_accuracy_per], columns)
        dataframe_fold.to_csv(os.path.join(path_fold, "out.csv"), decimal=",", sep=";", na_rep=" ", header=False,
                              quoting=csv.QUOTE_ALL)

        time_sec = time.strftime("%H:%M:%S", time.gmtime(t[0]["final_time"]))
        dataframe_time = pandas.DataFrame([t[0]["final_time"], time_sec], ["time", "time_sec"])
        dataframe_best_rule = pandas.DataFrame([best_rule["rule"], best_rule["accuracy"]],
                                               ["best_rule", "best_accuracy"])
        dataframe_info = pandas.concat([dataframe_time, dataframe_best_rule])
        dataframe_info.to_csv(os.path.join(path_fold, "fold_info.csv"), decimal=",", sep=";", na_rep=" ", header=False,
                              quoting=csv.QUOTE_ALL)



def classification_data(cfg, dataset, file_input, index, n_features, n_samples, path, x, y, n_patch=None, orientation=None):
    list_best_classifiers = list()
    list_result_classifier = list()

    for classifier in (
            sklearn.tree.DecisionTreeClassifier(random_state=1),
            sklearn.neighbors.KNeighborsClassifier(n_jobs=-1),
            sklearn.neural_network.MLPClassifier(random_state=1),
            sklearn.ensemble.RandomForestClassifier(random_state=1),
            sklearn.svm.SVC(random_state=1, probability=True)):
        classifier_name = classifier.__class__.__name__

        model = sklearn.model_selection.GridSearchCV(classifier, hyperparams[classifier_name], scoring="accuracy",
                                                     cv=cfg["fold"])
        model.fit(x, y)

        best_classifier = model.best_estimator_
        best_params = model.best_params_

        list_best_classifiers.append((classifier_name, best_classifier))


        data = [file_input, n_features, n_samples, n_patch, orientation]
        columns = ["file_input", "n_features", "n_samples", "n_patch", "orientation"]
        dataframe = pandas.DataFrame(data, columns)
        dataframe.to_csv(os.path.join(path, "info.csv"), decimal=",", sep=";", na_rep=" ", header=False,
                          quoting=csv.QUOTE_ALL)


        if n_patch and orientation:
            path_completed = os.path.join(path, classifier_name, orientation, str(n_patch), str(n_features))
        else:
            path_completed = os.path.join(path, classifier_name, str(n_features))
        pathlib.Path(path_completed).mkdir(parents=True, exist_ok=True)

        if n_patch and orientation:
            list_result_fold, list_time = data_has_patch(cfg, best_classifier, classifier_name, dataset, index, n_patch,
                                              path_completed, x, y)
        else:
            list_result_fold, list_time = data_no_patch(cfg, best_classifier, classifier_name, dataset, index, path, x, y)

        # save_mean_std(best_params, cfg, list_result_fold, list_time, n_features, n_samples, path)
        save(best_params, cfg, classifier_name, dataset, list_result_fold, list_time, path_completed)
        print("\n")
        list_result_classifier = list_result_classifier + list_result_fold
        # break

    # my_ensemble_classifier(cfg, dataset, list_result_classifier, n_features, n_samples, n_patch=n_patch, orientation=orientation)
    ensemble_classifier(cfg, dataset, index, list_best_classifiers, n_features, n_samples, path, x, y, n_patch=n_patch,
                        orientation=orientation)
