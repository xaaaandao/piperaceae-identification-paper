import os
import pathlib
import time

from classifier import get_best_classifier, get_list_classifiers, train_and_test, Classifier
from files import create_outfile_each_fold, create_outfile_mean_fold, plot_confusion_matrix, get_path_each_fold
from result import calculate_test, Result, convert_prob_to_label, sum_all_results, prod_all_results, max_all_results, \
    get_result_per_attribute_and_value, add_all_result
from samples import get_samples_with_patch, get_samples_train_and_test


def run_combine(cfg, dataset, list_result_classifier):
    list_classifier_name = list([getattr(result, "classifier") for result in list_result_classifier])
    list_classifier_name = list([getattr(classifier, "name") for classifier in list_classifier_name])
    if len(list(set(list_classifier_name))) < 2:
        raise ValueError("number invalid of classifier")

    all_classifiers = Classifier("+".join(l for l in set(list_classifier_name)), None, None, None, None)
    n_features = getattr(getattr(dataset, "list_data")[0], "n_features")
    list_elapsed_time = list([])
    list_result = list([])
    for fold in range(0, cfg["fold"]):
        print({fold}, getattr(all_classifiers, "name"))
        list_result_fold = get_result_per_attribute_and_value("fold", list_result_classifier, fold)
        y_true = getattr(list_result_fold[0], "y_true")
        start_time = time.time()
        result_max = Result(all_classifiers, fold, "max", None, max_all_results(list_result_fold), y_true)
        result_sum = Result(all_classifiers, fold, "sum", None, convert_prob_to_label(sum_all_results(list_result_fold)), y_true)
        result_prod = Result(all_classifiers, fold, "prod", None, convert_prob_to_label(prod_all_results(list_result_fold)), y_true)
        end_time = time.time()
        elapsed_time = end_time - start_time
        list_elapsed_time.append(elapsed_time)
        path_each_fold = get_path_each_fold(cfg, all_classifiers, dataset, str(n_features), str(fold))
        pathlib.Path(path_each_fold).mkdir(parents=True, exist_ok=True)
        create_outfile_each_fold(elapsed_time, (result_max, result_prod, result_sum), path_each_fold)
        plot_confusion_matrix(getattr(all_classifiers, "name"), getattr(dataset, "name"),
                              (result_max, result_prod, result_sum), path_each_fold)
        add_all_result(list_result, (result_max, result_prod, result_sum))
    path_mean = os.path.join(cfg["path_out"], getattr(dataset, "name"), getattr(all_classifiers, "name"), str(n_features))
    create_outfile_mean_fold(list_elapsed_time, list_result, path_mean)


def run(cfg, dataset, list_index):
    for data in getattr(dataset, "list_data"):
        x = getattr(data, "x")
        y = getattr(data, "y")
        list_result_classifier = list([])
        list_elapsed_time = list([])
        for classifier in get_list_classifiers():
            get_best_classifier(cfg, classifier, x, y)
            list_elapsed_time = list([])
            list_result_fold = list([])
            for index in list_index:
                print(getattr(index, "fold"), getattr(classifier, "name"), getattr(dataset, "name"), getattr(data, "x").shape, getattr(data, "y").shape)
                x_test, y_test, x_train, y_train = get_samples_train_and_test(index, x, y)
                start_time = time.time()
                y_pred = train_and_test(getattr(classifier, "best_classifier"), x_test, x_train, y_train)
                result_max, result_prod, result_sum = calculate_test(cfg, classifier, getattr(index, "fold"), y_pred, y_test)
                end_time = time.time()
                elapsed_time = end_time - start_time
                list_elapsed_time.append(elapsed_time)
                path_each_fold = get_path_each_fold(cfg, classifier, dataset, str(getattr(data, "n_features")), str(getattr(index, "fold")))
                pathlib.Path(path_each_fold).mkdir(parents=True, exist_ok=True)
                create_outfile_each_fold(elapsed_time, (result_max, result_prod, result_sum), path_each_fold)
                plot_confusion_matrix(getattr(classifier, "name"), getattr(dataset, "name"), (result_max, result_prod, result_sum), path_each_fold)
                add_all_result(list_result_fold, (result_max, result_prod, result_sum))
            path_mean = os.path.join(cfg["path_out"], getattr(dataset, "name"), getattr(classifier, "name"), str(getattr(data, "n_features")))
            create_outfile_mean_fold(list_elapsed_time, list_result_fold, path_mean)
            list_result_classifier = list_result_classifier + list_result_fold


def run_cnn(cfg, dataset, list_index):
    for data in getattr(dataset, "list_data"):
        x = getattr(data, "x")
        y = getattr(data, "y")
        list_result_classifier = list([])
        n_patch = getattr(getattr(dataset, "patch"), "n_patch")
        for classifier in get_list_classifiers()[4:]:
            get_best_classifier(cfg, classifier, x, y)
            list_result_fold = list([])
            list_elapsed_time = list([])
            for index in list_index:
                print(getattr(index, 'fold'), getattr(classifier, "name"), getattr(dataset, "name"),
                      getattr(data, "x").shape, getattr(data, "y").shape)
                index_test = getattr(index, "index_test")
                index_train = getattr(index, "index_train")
                x_test, y_test = get_samples_with_patch(x, y, index_test, n_patch)
                x_train, y_train = get_samples_with_patch(x, y, index_train, n_patch)
                start_time = time.time()
                y_pred = train_and_test(getattr(classifier, "best_classifier"), x_test, x_train, y_train)
                result_max, result_prod, result_sum = calculate_test(cfg, classifier, getattr(index, "fold"), y_pred, y_test, n_patch=n_patch)
                end_time = time.time()
                elapsed_time = end_time - start_time
                list_elapsed_time.append(elapsed_time)
                path_each_fold = get_path_each_fold(cfg, classifier, dataset, str(data.n_features), str(getattr(index, "fold")))
                pathlib.Path(path_each_fold).mkdir(parents=True, exist_ok=True)
                create_outfile_each_fold(elapsed_time, (result_max, result_prod, result_sum), path_each_fold)
                plot_confusion_matrix(getattr(classifier, "name"), getattr(dataset, "name"), (result_max, result_prod, result_sum), path_each_fold)
                add_all_result(list_result_fold, (result_max, result_prod, result_sum))
                print("+++++++++++++++++++++++++++++++++")
            path_mean = os.path.join(cfg["path_out"], getattr(dataset, "name"), getattr(classifier, "name"), str(data.n_features))
            create_outfile_mean_fold(list_elapsed_time, list_result_fold, path_mean)
            list_result_classifier = list_result_classifier + list_result_fold
            print("---------------------------------")
        break

