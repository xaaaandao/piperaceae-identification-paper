import collections
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from arrays import split_dataset
from result import Result
from save import save_csv_transpose


hyper = {
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
        'n_estimators': [200, 400, 600],
        'max_features': ['sqrt', 'log2'],
        'criterion': ['gini', 'entropy']
    },
    'SVC': {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }
}

class Fold:
    def __init__(self, dataset, fold, idx_train, idx_test):
        self.best_f1 = None
        self.best_accuracy = None
        self.best_classifier = None
        self.count_train = None
        self.count_test = None
        self.dataset = dataset
        self.fold = fold
        self.idx_train = idx_train
        self.idx_test = idx_test
        self.results = list()
        self.total_test = 0
        self.total_train = 0
        self.total_test_no_patch = 0
        self.total_train_no_patch = 0
        self.x_test = list()
        self.x_train = list()
        self.y_pred_proba = list()
        self.y_test = list()
        self.y_train = list()

    def run(self, backend, classifier, **kwargs):
        self.best_classifier = GridSearchCV(classifier, hyper[classifier.__class__.__name__], **kwargs)

        with joblib.parallel_backend(backend, n_jobs=kwargs["n_jobs"]):
            self.best_classifier.fit(self.dataset.x, self.dataset.y)

        if isinstance(self.best_classifier.best_estimator_, SVC):
            params = dict(probability=True)
            self.best_classifier.best_estimator_.set_params(**params)

        self.x_train, self.y_train = split_dataset(self.idx_train, self.dataset.n_features, self.dataset.patch, self.dataset.x, self.dataset.y)
        self.x_test, self.y_test = split_dataset(self.idx_test, self.dataset.n_features, self.dataset.patch, self.dataset.x, self.dataset.y)

        self.count_train = collections.Counter(self.y_train)
        self.count_test = collections.Counter(self.y_test)
        self.total_test = np.sum(list(self.count_test.values()))
        self.total_train = np.sum(list(self.count_train.values()))
        self.total_test_no_patch = self.total_test / self.dataset.patch
        self.total_train_no_patch = self.total_train / self.dataset.patch

        logging.info("Train: %s" % self.count_train)
        logging.info("Test: %s" % self.count_test)

        logging.info("Total train: %s" % self.total_train_no_patch)
        logging.info("Total test: %s" % self.total_test_no_patch)

        self.a()

        self.best_classifier.best_estimator_.fit(self.x_train, self.y_train)
        self.y_pred_proba = self.best_classifier.best_estimator_.predict_proba(self.x_test)

        self.results = [Result(self.dataset, rule, self.y_pred_proba, self.y_test) for rule in ["sum", "max", "mult"]]
        for result in self.results:
            n_test, n_labels = self.y_pred_proba.shape
            result.evaluate(n_test, n_labels)

        self.best_f1 = max(self.results, key=lambda x: x.f1)
        self.best_accuracy = max(self.results, key=lambda x: x.accuracy)
        logging.info("Best result F1: %s Rule: %s" % (str(self.best_f1.f1), self.best_f1.rule))
        logging.info("Best result accuracy: %s Rule: %s" % (str(self.best_accuracy.accuracy), self.best_f1.rule))

    def save(self, output):
        self.save_best(output)
        self.save_count(output)
        self.save_fold(output)
        self.save_idx(output)
        self.save_results(output)

    def save_best(self, output):
        output_dir = os.path.join(output, "best")
        os.makedirs(output_dir, exist_ok=True)

        self.save_best_classifier(output_dir)
        self.save_best_results(output_dir)

    def save_best_classifier(self, output):
        self.save_best_classifier_cv_results(output)
        self.save_best_classifier_pkl(output)

    def save_best_classifier_pkl(self, output):
        filename = os.path.join(output, "fold-%d-best_classifier.pkl" % self.fold)
        logging.info("saving %s" % filename)

        try:
            with open(filename, "wb") as file:
                joblib.dump(self.best_classifier, file, compress=3)
            file.close()
        except FileExistsError:
            logging.warning("problems in save model (%s)" % filename)

    def save_best_classifier_cv_results(self, output):
        filename = os.path.join(output, "fold-%d-best_classifier.csv" % self.fold)

        df = pd.DataFrame(self.best_classifier.cv_results_)
        df.to_csv(filename, index=False, header=True, sep=";", quoting=2, encoding="utf-8")
        logging.info("saving %s" % filename)

    def save_best_results(self, output):
        filename = os.path.join(output, "fold-%d-best_results.csv" % self.fold)
        data = {
            "best_f1": [self.best_f1.f1],
            "best_f1_rule": [self.best_f1.rule],
            "best_accuracy": [self.best_accuracy.accuracy],
            "best_accuracy_rule": [self.best_accuracy.rule]
        }
        save_csv_transpose(data, filename)

    def save_results(self, output):
        output_dir = os.path.join(output, "results")
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.join(output_dir, "fold-%d-results.csv" % (self.fold))
        df = pd.DataFrame([result.to_dict() for result in self.results])
        df.to_csv(filename, sep=";", quoting=2, index=False, header=True)
        logging.info("saving %s" % filename)

        for result in self.results:
            output_dir = os.path.join(output, "results", result.rule)
            os.makedirs(output_dir, exist_ok=True)

            result.save_predictions(self.fold, output_dir)
            result.save_confusion_matrix(self.fold, output_dir)
            result.save_classification_report(self.fold, output_dir)
            result.save_topk(self.fold, output_dir, self.total_test_no_patch)
            result.save_tp(self.count_test, self.fold, output_dir, self.dataset.patch, self.total_test_no_patch)

    def save_fold(self, output):
        filename = os.path.join(output, "fold-%d.csv" % self.fold)
        data = {
            "total_test": [self.total_test],
            "total_train": [self.total_train],
            "total_test_no_patch": [self.total_test_no_patch],
            "total_train_no_patch": [self.total_train_no_patch],
        }
        save_csv_transpose(data, filename)

    def save_idx(self, output):
        output_dir = os.path.join(output, "idx")
        os.makedirs(output_dir, exist_ok=True)

        self.save_idx_train(output_dir)
        self.save_idx_test(output_dir)

    def save_idx_train(self, output: str):
        filename = os.path.join(output, "fold-%d-idx_train.npy" % self.fold)
        np.save(filename, self.idx_train)
        logging.info("saving %s" % filename)

    def save_idx_test(self, output_dir: str):
        filename = os.path.join(output_dir, "fold-%d-idx_test.npy" % self.fold)
        np.save(filename, self.idx_test)
        logging.info("saving %s" % filename)

    def save_count(self, output):
        filename = os.path.join(output, "fold-%d-count.csv" % self.fold)
        data = []
        for l in self.dataset.levels:
            data.append({"label": l.label,
                         "specific_epithet": l.specific_epithet,
                         "count_train": self.count_train[l.label] / self.dataset.patch,
                         "count_test": self.count_test[l.label] / self.dataset.patch,
            })
        df = pd.DataFrame(data)
        df.to_csv(filename, sep=";", quoting=2, index=False)

    def a(self):
        if self.dataset.x_augmented.shape[0] > 0:
            select_files = self.dataset.filenames[self.idx_train]
            print(collections.Counter(select_files))
            print(len(select_files), len(self.idx_train))
            print(len(np.unique(select_files)))
            # print(self.dataset.x_augmented.shape)
            # b = self.dataset.x_augmented[np.isin(self.dataset.x_augmented[:, -1], select_files)]
            # print(b.shape)
            # t = 0
            # for sf in select_files:
            #     c = self.dataset.x_augmented[np.isin(self.dataset.x_augmented[:, -1], sf)]
            #     print(sf, c.shape)
            #     t = t + c.shape[0]
            # print(t)
            import sys
            sys.exit()

