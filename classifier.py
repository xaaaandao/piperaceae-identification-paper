import dataclasses
import os
import pathlib
import time

import sklearn.ensemble
import warnings

from output import save_fold
from result import calculate_test
from samples import get_samples_with_patch

warnings.simplefilter("ignore", category=sklearn.exceptions.ConvergenceWarning)


def classifier_patch(cfg, best_classifier, classifier_name, dataset, fold, index_test,
                     index_train, list_result_fold, list_time, n_patch, path_classifier, pca, x, y):
    x_train, y_train = get_samples_with_patch(x, y, index_train, n_patch)
    x_test, y_test = get_samples_with_patch(x, y, index_test, n_patch)

    print(fold, classifier_name, x_train.shape, x_test.shape)

    start_time = time.time()
    best_classifier.fit(x_train, y_train)
    y_pred = best_classifier.predict_proba(x_test)
    end_time = time.time()

    path_fold = os.path.join(path_classifier, str(n_patch), str(pca), str(fold))
    pathlib.Path(path_fold).mkdir(parents=True, exist_ok=True)

    result_max_rule, result_prod_rule, result_sum_rule = calculate_test(cfg, fold, y_pred, y_test, n_patch=n_patch)

    final_time = end_time - start_time

    list_result_fold.append(result_max_rule)
    list_result_fold.append(result_prod_rule)
    list_result_fold.append(result_sum_rule)
    list_time.append(final_time)

    save_fold(classifier_name, dataset, final_time, (result_max_rule, result_prod_rule, result_sum_rule),
              path_fold)


def classifier_no_patch(cfg, best_classifier, classifier_name, dataset, fold, index_test,
                        index_train, list_result_fold, list_time, path_classifier, pca, x, y):
    x_train, y_train = x[index_train], y[index_train]
    x_test, y_test = x[index_test], y[index_test]

    print(fold, classifier_name, x_train.shape, x_test.shape)

    start_time = time.time()
    best_classifier.fit(x_train, y_train)
    y_pred = best_classifier.predict_proba(x_test)
    end_time = time.time()

    path_fold = os.path.join(path_classifier, str(pca), str(fold))
    pathlib.Path(path_fold).mkdir(parents=True, exist_ok=True)

    result_max_rule, result_prod_rule, result_sum_rule = calculate_test(cfg, fold, y_pred, y_test)

    final_time = end_time - start_time

    list_result_fold.append(result_max_rule)
    list_result_fold.append(result_prod_rule)
    list_result_fold.append(result_sum_rule)
    list_time.append(final_time)

    save_fold(classifier_name, dataset, final_time, (result_max_rule, result_prod_rule, result_sum_rule), path_fold)


@dataclasses.dataclass
class Classifier:
    name: str
    classifier: None
    params: dict
    best_classifier: None
    best_params: dict
