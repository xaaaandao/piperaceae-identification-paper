import collections
import dataclasses
import itertools
import logging

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from dataset import Dataset
from result import Result
from predict import Predict
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
    def __init__(self, x: np.ndarray, y: np.ndarray,
                 filenames: list[str], patch: int):
        self.x: np.ndarray = x
        self.y: np.ndarray = y
        self.filenames: list[str] = filenames
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
        self.test: Data = None
        self.train: Data = None
        self.x_aug = []
        self.y_aug = []
        self.y_pred_proba = []

    def split_fold(self, idxs):
        rows_to_get = []
        for idx in idxs:
            start_row = idx * self.dataset.patch
            end_row = start_row + self.dataset.patch
            rows_to_get.extend(range(start_row, end_row))

        return self.dataset.x[rows_to_get], self.dataset.y[rows_to_get], self.dataset.filenames[rows_to_get]

    def run(self, backend, classifier, data_augmentations, **kwargs):
        x_train, y_train, filenames_train = self.split_fold(self.idx.idx_train)
        x_test, y_test, filenames_test = self.split_fold(self.idx.idx_test)

        self.train = Data(x_train, y_train, filenames_train, self.dataset.patch)
        self.test = Data(x_test, y_test, filenames_test, self.dataset.patch)

        if len(data_augmentations) > 0:
            train_x_shape = self.train.x.shape
            self.find_train_data_augmentation(data_augmentations)
            self.merge_train_data_augmentation()

            logging.info("x_train COM data augmentation: %s" % str(self.train.x.shape))
            logging.info("y_train COM data augmentation: %s" % str(self.train.y.shape))

            if train_x_shape[0] + self.x_aug.shape[0] != self.train.x.shape[0]:
                raise SystemExit("shape not match")

            if len(np.setdiff1d(self.train.filenames, self.filenames_aug)) == 0 and len(np.setdiff1d(self.filenames_aug, self.train.filenames)):
                raise SystemExit("filenames not match")
            
        scaler = StandardScaler()
        self.train.x = scaler.fit_transform(self.train.x)
        self.test.x = scaler.fit_transform(self.test.x)

        self.best_classifier = GridSearchCV(classifier, hyper[classifier.__class__.__name__], **kwargs)

        with joblib.parallel_backend(backend, n_jobs=kwargs["n_jobs"]):
            self.best_classifier.fit(self.train.x, self.train.y)

        if isinstance(self.best_classifier.best_estimator_, SVC):
            params = dict(probability=True)
            self.best_classifier.best_estimator_.set_params(**params)

        self.best_classifier.best_estimator_.fit(self.train.x, self.train.y)
        y_pred_proba = self.best_classifier.best_estimator_.predict_proba(self.test.x)

        self.predicts = [Predict(self.dataset.patch, r, y_pred_proba, self.test.y) for r in self.rules]
        self.results = [Result(self.dataset.levels, p) for p in self.predicts]
        self.best_result = BestResult(self.results)

    def find_train_data_augmentation(self, data_augmentations):
        datas = []
        for d in data_augmentations:
            data_aug = d.data

            if "cut-mix" in d.input_dir:
                filenames = ["+".join(c) for c in itertools.combinations(self.train.filenames, 2)]
            else:
                filenames = self.train.filenames

            logging.info("data augmentation (%s) shape: %s" % (d.input_dir, str(data_aug.shape)))
            mask = np.isin(data_aug[:, -1], filenames)
            datas.append(data_aug[mask])

        data_aug = np.vstack(datas)
        logging.info("SELECTED data_aug shape: %s" % str(data_aug.shape))

        self.x_aug = data_aug[:, :-2]
        self.x_aug = self.x_aug.astype(float)

        self.y_aug = data_aug[:, -2]
        self.y_aug = self.y_aug.astype(float).astype(np.int16)

        self.filenames_aug = data_aug[:, -1]

        logging.info("x_augmented shape: %s" % str(self.x_aug.shape))
        logging.info("y_augmented shape: %s" % str(self.y_aug.shape))

    def merge_train_data_augmentation(self):
        self.train.x = np.concatenate((self.train.x, self.x_aug), axis=0)
        self.train.y = np.concatenate((self.train.y, self.y_aug), axis=0)


    def to_dict_data_count_level(self):
        if self.x_aug is not None and len(self.x_aug) > 0:
            count = collections.Counter(self.y_aug)
            # TODO pega o patch do data_aug
            return {
                "levels": [l.name for l in sorted(self.dataset.levels, key=lambda x: x.label)],
                "count_train": self.train.to_level_count(self.dataset.levels),
                "after_count_train": [count[l.label] // self.dataset.patch for l in sorted(self.dataset.levels, key=lambda x:x.label)],
                "count_test": self.test.to_level_count(self.dataset.levels)
            }
        return {
            "levels": [l.name for l in sorted(self.dataset.levels, key=lambda x: x.label)],
            "count_train": self.train.to_level_count(self.dataset.levels),
            "count_test": self.test.to_level_count(self.dataset.levels)
        }

    def to_dict_data_total(self):
        return {
            "train_test": ["train", "test"],
            "total": [self.train.total, self.test.total],
            "total_no_patch": [self.train.total_no_patch, self.test.total_no_patch],
        }

