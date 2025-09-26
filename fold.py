import collections
import itertools
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from arrays import split_dataset
from result import Result, BestResult
from save import save_csv_transpose, SaveFold

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

class Fold:
    def __init__(self, dataset, fold, idx_train, idx_test):
        # self.best_f1 = None
        # self.best_accuracy = None
        self.best_classifier = None
        self.best_result = None
        self.count_train = None
        self.count_test = None
        self.dataset = dataset
        self.fold = fold
        self.idx_train = idx_train
        self.idx_test = idx_test
        self.results = list()
        self.s = None
        self.total_test = 0
        self.total_train = 0
        self.total_test_no_patch = 0
        self.total_train_no_patch = 0
        self.x_aug = np.array([])
        self.x_test = np.array([])
        self.x_train = np.array([])
        self.y_aug = np.array([])
        self.y_pred_proba = np.array([])
        self.y_test = np.array([])
        self.y_train = np.array([])

    def run(self, backend, classifier, data_augmentations, **kwargs):
        self.x_train, self.y_train = split_dataset(self.idx_train, self.dataset.n_features, self.dataset.patch, self.dataset.x, self.dataset.y)
        self.x_test, self.y_test = split_dataset(self.idx_test, self.dataset.n_features, self.dataset.patch, self.dataset.x, self.dataset.y)

        logging.info("x_train: %s" % str(self.x_train.shape))
        logging.info("y_train: %s" % str(self.y_train.shape))

        self.get_train_data_augmentation(data_augmentations)

        if self.x_aug is not None and self.x_aug.shape[0] > 0:
            self.x_train = np.concatenate((self.x_train, self.x_aug), axis=0)
            self.y_train = np.concatenate((self.y_train, self.y_aug), axis=0)
            logging.info("x_train COM data augmentation: %s" % str(self.x_train.shape))
            logging.info("y_train COM data augmentation: %s" % str(self.y_train.shape))

        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.fit_transform(self.x_test)

        self.count_train = collections.Counter(self.y_train)
        self.count_test = collections.Counter(self.y_test)
        self.total_test = np.sum(list(self.count_test.values()))
        self.total_train = np.sum(list(self.count_train.values()))
        self.total_test_no_patch = self.total_test / self.dataset.patch
        self.total_train_no_patch = self.total_train / self.dataset.patch

        logging.info("Fold: %d" % self.fold)
        logging.info("Train: %s" % self.count_train)
        logging.info("Test: %s" % self.count_test)
        logging.info("Total train: %s" % self.total_train_no_patch)
        logging.info("Total test: %s" % self.total_test_no_patch)

        self.best_classifier = GridSearchCV(classifier, hyper[classifier.__class__.__name__], **kwargs)

        with joblib.parallel_backend(backend, n_jobs=kwargs["n_jobs"]):
            self.best_classifier.fit(self.x_train, self.y_train)

        if isinstance(self.best_classifier.best_estimator_, SVC):
            params = dict(probability=True)
            self.best_classifier.best_estimator_.set_params(**params)

        self.best_classifier.best_estimator_.fit(self.x_train, self.y_train)
        self.y_pred_proba = self.best_classifier.best_estimator_.predict_proba(self.x_test)

        self.results = [Result(self.dataset, rule, self.y_pred_proba, self.y_test) for rule in ["sum", "max", "mult"]]

        for result in self.results:
            n_test, n_labels = self.y_pred_proba.shape
            result.evaluate(n_test, n_labels)

        self.best_result = BestResult(self.results)

    def save(self, output):
        self.s = SaveFold(self, output)

    def get_train_data_augmentation(self, data_augmentations):
        if len(data_augmentations) > 0:
            data_aug = [d.data for d in data_augmentations]
            data_aug = np.array(list(itertools.chain(*data_aug)))
            logging.info("merge data augmentations: %s" % str(data_aug.shape))

            filenames, idx = np.unique(self.dataset.filenames, return_index=True)
            filenames = filenames[np.argsort(idx)]
            filenames = filenames[self.idx_train]

            features = data_aug[np.isin(data_aug[:, -1], filenames)]
            features = np.vstack(features)
            self.x_aug = features[:, :-2]
            self.x_aug = self.x_aug.astype(float)
            self.y_aug = features[:, -2]
            self.y_aug = self.y_aug.astype(float).astype(np.int16)
            logging.info("x_augmented shape: %s" % str(self.x_aug.shape))
            logging.info("y_augmented shape: %s" % str(self.y_aug.shape))

