import collections
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from dataset import Dataset
from result import Result
from predict import Predict
from save import save_csv
from best import BestResult

hyper = {
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
        "n_estimators": [200, 400, 600],
        "max_features": ["sqrt", "log2"],
        "criterion": ["gini", "entropy"]
    },
    "SVC": {
        "kernel": ["linear", "poly", "rbf", "sigmoid"]
    }
}

class IndexTrainTest:
    def __init__(self, idx):
        self.idx_train = idx[0]
        self.idx_test = idx[1]


class Data:
    def __init__(self, x, y, filenames, patch):
        self.x = x
        self.y = y
        self.filenames = filenames
        logging.info("x.shape: %s y.shape: %s" % (self.x.shape, self.y.shape))

        self.count = collections.Counter(self.y)
        logging.info("count: %s" % self.count)

        self.patch = patch
        self.total = np.sum(list(self.count.values()))
        self.total_no_patch = self.total // self.patch
        logging.info("patch: %d total: %d total no_patch: %s" % (self.patch, self.total, self.total_no_patch))

    def to_level_count(self, levels):
        return [self.count[l.label] // self.patch for l in sorted(levels, key=lambda x:x.label)]

class Fold:
    def __init__(self, dataset: Dataset, fold, idx):
        self.best_classifier = None
        self.best_result = None
        self.dataset = dataset
        self.fold = fold
        self.idx = IndexTrainTest(idx)
        self.results = list
        self.rules = ["sum", "max", "mult"]
        self.y_pred_proba = []

    def split_fold(self, idxs):
        rows_to_get = []
        for idx in idxs:
            start_row = idx * self.dataset.patch
            end_row = start_row + self.dataset.patch
            rows_to_get.extend(range(start_row, end_row))

        return self.dataset.x[rows_to_get], self.dataset.y[rows_to_get], self.dataset.filenames[rows_to_get]


    def run(self, backend, classifier, output, **kwargs):
        output_dir = os.path.join(output, "fold-%d" % self.fold)
        os.makedirs(output_dir, exist_ok=True)

        logging.info("fold: %d clf: %s" % (self.fold, classifier.__class__.__name__))
        x_train, y_train, filenames_train = self.split_fold(self.idx.idx_train)
        x_test, y_test, filenames_test = self.split_fold(self.idx.idx_test)

        self.train = Data(x_train, y_train, filenames_train, self.dataset.patch)
        self.test = Data(x_test, y_test, filenames_test, self.dataset.patch)

        self.best_classifier = GridSearchCV(classifier, hyper[classifier.__class__.__name__], **kwargs)

        with joblib.parallel_backend(backend, n_jobs=kwargs["n_jobs"]):
            self.best_classifier.fit(self.train.x, self.train.y)

        if isinstance(self.best_classifier.best_estimator_, SVC):
            params = dict(probability=True)
            self.best_classifier.best_estimator_.set_params(**params)

        self.best_classifier.best_estimator_.fit(self.train.x, self.train.y)
        y_pred_proba = self.best_classifier.best_estimator_.predict_proba(self.test.x)

        predicts = [Predict(self.dataset.patch, r, y_pred_proba, self.test.y) for r in self.rules]
        self.results = [Result(self.fold, self.dataset.levels, output_dir, p) for p in predicts]
        self.best_result = BestResult(self.results)

        self.save(output_dir)

    def save(self, output):
        self.save_best(output)
        self.save_idx(output)
        self.save_data(output)

    def save_data(self, output):
        self.save_data_count_level(output)
        self.save_data_total(output)

    def save_data_count_level(self, output):
        count_train = self.train.to_level_count(self.dataset.levels)
        count_test = self.test.to_level_count(self.dataset.levels)
        data = {
            "levels": [l.name for l in sorted(self.dataset.levels, key=lambda x: x.label)],
            "count_train": count_train,
            "count_test": count_test,
        }
        filename = os.path.join(output, "fold-%d-count.csv" % self.fold)
        df = pd.DataFrame(data)
        save_csv(df, filename)

    def save_data_total(self, output):
        data = {
            "train_test": ["train", "test"],
            "total": [self.train.total, self.test.total],
            "total_no_patch": [self.train.total_no_patch, self.test.total_no_patch],
        }
        filename = os.path.join(output, "fold-%d.csv" % self.fold)
        df = pd.DataFrame(data)
        save_csv(df, filename)

    def save_best(self, output):
        output_dir = os.path.join(output, "best")
        os.makedirs(output_dir, exist_ok=True)
        self.save_best_classifier(output_dir)
        self.best_result.save(output_dir)

    def save_best_classifier(self, output):
        self.save_best_classifier_cv_results(output)
        self.save_best_classifier_pkl(output)

    def save_best_classifier_pkl(self, output):
        filename = os.path.join(output, "best_classifier.pkl")
        logging.info("saved %s" % filename)

        try:
            with open(filename, "wb") as file:
                joblib.dump(self.best_classifier, file, compress=3)
            file.close()
        except FileExistsError:
            logging.warning("problems in save model (%s)" % filename)

    def save_best_classifier_cv_results(self, output):
        filename = os.path.join(output, "best_classifier_cv_results.csv")
        data = self.best_classifier.cv_results_
        df = pd.DataFrame(data)
        save_csv(df, filename, header=True, index=False)

    def save_idx_test(self, output):
        filename = os.path.join(output, "fold-%d-idx_test.npy" % self.fold)
        np.save(filename, self.idx.idx_test)
        logging.info("saved idx_test: %s" % filename)

    def save_idx(self, output):
        output_dir = os.path.join(output, "idx")
        os.makedirs(output_dir, exist_ok=True)
        self.save_idx_train(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        self.save_idx_test(output_dir)

    def save_idx_train(self, output):
        filename = os.path.join(output, "fold-%d-idx_train.npy" % self.fold)
        np.save(filename, self.idx.idx_train)
        logging.info("saved idx_train: %s" % filename)

