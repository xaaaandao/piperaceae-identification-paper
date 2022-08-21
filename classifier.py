import os
import pathlib
import sklearn.ensemble
import sklearn.exceptions
import sklearn.neighbors
import sklearn.neural_network
import sklearn.preprocessing
import sklearn.svm
import time
import warnings

from output import save_fold, save_mean_std
from result import calculate_test, get_result_per_attribute_and_value, Result, convert_prob_to_label, max_all_results, \
    sum_all_results, prod_all_results
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


def data_has_patch(cfg, best_classifier, classifier_name, dataset, index, list_result_fold, list_time, n_patch,
                   path_classifier, x, x_surf, y, y_surf):
    for fold, (index_train, index_test) in enumerate(index.split(x_surf, y_surf)):
        x_train, y_train = get_samples_with_patch(x, y, index_train, n_patch)
        x_test, y_test = get_samples_with_patch(x, y, index_test, n_patch)

        print(fold, classifier_name, x_train.shape, x_test.shape)

        start_time = time.time()
        best_classifier.fit(x_train, y_train)
        y_pred = best_classifier.predict_proba(x_test)
        end_time = time.time()

        path_fold = os.path.join(path_classifier, str(fold))
        pathlib.Path(path_fold).mkdir(parents=True, exist_ok=True)

        result_max_rule, result_prod_rule, result_sum_rule = calculate_test(cfg, fold, y_pred, y_test, n_patch=n_patch)

        final_time = end_time - start_time

        list_result_fold.append(result_max_rule)
        list_result_fold.append(result_prod_rule)
        list_result_fold.append(result_sum_rule)
        list_time.append(final_time)

        save_fold(classifier_name, dataset, final_time, (result_max_rule, result_prod_rule, result_sum_rule),
                  path_fold)


def data_no_patch(cfg, best_classifier, classifier_name, dataset, index, list_result_fold, list_time, path_classifier,
                  x, y):
    for fold, (index_train, index_test) in enumerate(index.split(x, y)):
        x_train, y_train = x[index_train], y[index_train]
        x_test, y_test = x[index_test], y[index_test]

        print(fold, classifier_name, x_train.shape, x_test.shape)

        start_time = time.time()
        best_classifier.fit(x_train, y_train)
        y_pred = best_classifier.predict_proba(x_test)
        end_time = time.time()

        path_fold = os.path.join(path_classifier, str(fold))
        pathlib.Path(path_fold).mkdir(parents=True, exist_ok=True)

        result_max_rule, result_prod_rule, result_sum_rule = calculate_test(cfg, fold, y_pred, y_test)

        final_time = end_time - start_time

        list_result_fold.append(result_max_rule)
        list_result_fold.append(result_prod_rule)
        list_result_fold.append(result_sum_rule)
        list_time.append(final_time)

        save_fold(classifier_name, dataset, final_time, (result_max_rule, result_prod_rule, result_sum_rule), path_fold)


def my_ensemble_classifier(cfg, dataset, list_result_classifier, n_features, n_samples, n_patch=None, orientation=None):
    classifier_name = "MyEnsembleClassifier"
    list_result_mean = list()
    if n_patch and orientation:
        path_classifier = os.path.join(cfg["path_out"], dataset, classifier_name, orientation, str(n_patch),
                                       str(n_features))
    else:
        path_classifier = os.path.join(cfg["path_out"], dataset, classifier_name, str(n_features))
    pathlib.Path(path_classifier).mkdir(parents=True, exist_ok=True)

    list_time = list()
    for fold in range(0, cfg["fold"]):
        list_result_fold = get_result_per_attribute_and_value("fold", list_result_classifier, fold)

        if len(list_result_fold) < 2:
            raise ValueError("number invalid of classifier")

        print(fold, classifier_name)

        y_true = getattr(list_result_fold[0], "y_true")
        start_time = time.time()
        result_max_rule = Result(fold, "max", None, max_all_results(list_result_fold), y_true)
        result_sum_rule = Result(fold, "sum", None,
                            convert_prob_to_label(sum_all_results(list_result_fold)), y_true)
        result_prod_rule = Result(fold, "prod", None,
                            convert_prob_to_label(prod_all_results(list_result_fold)), y_true)
        end_time = time.time()

        final_time = end_time - start_time

        list_result_mean.append(result_max_rule)
        list_result_mean.append(result_sum_rule)
        list_result_mean.append(result_prod_rule)
        list_time.append(final_time)

        path_fold = os.path.join(path_classifier, str(fold))
        pathlib.Path(path_fold).mkdir(parents=True, exist_ok=True)
        save_fold(classifier_name, dataset, final_time, (result_max_rule, result_prod_rule, result_sum_rule), path_fold)

    save_mean_std(None, cfg, list_result_mean, list_time, n_features, n_samples, path_classifier)


def ensemble_classifier(cfg, dataset, index, list_best_classifiers, n_features, n_samples, path, x, y, n_patch=None,
                        orientation=None, x_surf=None, y_surf=None):
    classifier = sklearn.ensemble.VotingClassifier(estimators=list_best_classifiers, voting="hard")
    classifier_name = classifier.__class__.__name__

    list_result_fold = list()
    list_time = list()

    if n_patch and orientation:
        data_has_patch(cfg, classifier, classifier_name, dataset, index, list_result_fold, list_time, n_patch,
                       path, x, x_surf, y, y_surf)
    else:
        data_no_patch(cfg, classifier, classifier_name, dataset, index, list_result_fold, list_time, path, x,
                      y)
    save_mean_std(None, cfg, list_result_fold, list_time, n_features, n_samples, path)


def classification_data(cfg, dataset, index, n_features, n_samples, x, y, n_patch=None, orientation=None, x_surf=None,
                        y_surf=None):
    list_best_classifiers = list()
    list_result_classifier = list()

    # for classifier in (
    #                           sklearn.tree.DecisionTreeClassifier(random_state=1),
    #                           sklearn.neighbors.KNeighborsClassifier(n_jobs=-1),
    #                           sklearn.neural_network.MLPClassifier(random_state=1),
    #                           sklearn.ensemble.RandomForestClassifier(random_state=1),
    #                           sklearn.svm.SVC(random_state=1, probability=True))[3:]:
    for classifier in (
            sklearn.tree.DecisionTreeClassifier(random_state=1), sklearn.neighbors.KNeighborsClassifier(n_jobs=-1)):
        classifier_name = classifier.__class__.__name__

        model = sklearn.model_selection.GridSearchCV(classifier, hyperparams[classifier_name], scoring="accuracy",
                                                     cv=cfg["fold"])
        model.fit(x, y)

        best_classifier = model.best_estimator_
        best_params = model.best_params_

        list_best_classifiers.append((classifier_name, best_classifier))
        list_result_fold = list()
        list_time = list()

        if n_patch and orientation:
            path = os.path.join(cfg["path_out"], dataset, classifier_name, orientation, str(n_patch), str(n_features))
        else:
            path = os.path.join(cfg["path_out"], dataset, classifier_name, str(n_features))
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        if n_patch and orientation:
            data_has_patch(cfg, best_classifier, classifier_name, dataset, index, list_result_fold, list_time, n_patch,
                           path, x, x_surf, y, y_surf)
        else:
            data_no_patch(cfg, best_classifier, classifier_name, dataset, index, list_result_fold, list_time, path, x,
                          y)

        save_mean_std(best_params, cfg, list_result_fold, list_time, n_features, n_samples, path)
        list_result_classifier = list_result_classifier + list_result_fold

    my_ensemble_classifier(cfg, dataset, list_result_classifier, n_features, n_samples, n_patch=n_patch, orientation=orientation)
    # ensemble_classifier(cfg, dataset, index, list_best_classifiers, n_features, n_samples, path, x, y, n_patch=n_patch, orientation=orientation, x_surf=x_surf, y_surf=y_surf)
