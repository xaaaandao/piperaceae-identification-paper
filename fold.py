import collections
import logging

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from arrays import split_dataset
from dataset import Dataset
from result import Result, Predict

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


class Fold:
    def __init__(self, dataset: Dataset, fold, idx):
        self.best_classifier = None
        self.dataset = dataset
        self.fold = fold
        self.idx = IndexTrainTest(idx)
        self.results = list
        self.rules = ["sum", "max", "mult"]
        self.y_pred_proba = []

    def run(self, backend, classifier, **kwargs):
        logging.info("fold: %d clf: %s" % (self.fold, classifier.__class__.__name__))
        x_train, y_train, filenames_train = split_dataset(self.idx.idx_train, self.dataset.qtd_features, self.dataset.patch, self.dataset.x, self.dataset.y, self.dataset.filenames)
        x_test, y_test, filenames_test = split_dataset(self.idx.idx_test, self.dataset.qtd_features, self.dataset.patch, self.dataset.x, self.dataset.y, self.dataset.filenames)

        train = Data(x_train, y_train, filenames_train, self.dataset.patch)
        test = Data(x_test, y_test, filenames_test, self.dataset.patch)

        self.best_classifier = GridSearchCV(classifier, hyper[classifier.__class__.__name__], **kwargs)

        with joblib.parallel_backend(backend, n_jobs=kwargs["n_jobs"]):
            self.best_classifier.fit(train.x, train.y)

        if isinstance(self.best_classifier.best_estimator_, SVC):
            params = dict(probability=True)
            self.best_classifier.best_estimator_.set_params(**params)

        self.best_classifier.best_estimator_.fit(train.x, train.y)
        y_pred_proba = self.best_classifier.best_estimator_.predict_proba(test.x)

        predicts = [Predict(self.dataset.patch, r, y_pred_proba, test.y) for r in self.rules]
        self.results = [Result(self.dataset.levels, p) for p in predicts]
        # self.best_result = BestResult(self.results)