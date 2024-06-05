import collections
import dataclasses
import logging
import timeit
from typing import Any

import numpy as np

from arrays import split_dataset
from dataset import Dataset
from predict import Predict
from result import Result


class Fold:
    # fold: int = dataclasses.field(init=True)
    # idx_train: Any = dataclasses.field(init=False)
    # idx_test: Any = dataclasses.field(init=False)
    # x: np.ndarray = dataclasses.field(init=True)
    # y: np.ndarray = dataclasses.field(init=True)
    # result: Result = dataclasses.field(init=False, default_factory=Result)

    def __init__(self, fold: int, idx: Any, x: np.ndarray, y: np.ndarray, result: Result=None):
        self.fold = fold
        self.idx_train = idx[0]
        self.idx_test = idx[1]
        self.result = result
        self.x = x
        self.y = y

    def run(self, classifier: Any, dataset: Dataset):
        x_train, y_train = split_dataset(self.idx_train, dataset.count_features, dataset.image.patch, self.x, self.y)
        x_test, y_test = split_dataset(self.idx_test, dataset.count_features, dataset.image.patch, self.x, self.y)

        count_train = collections.Counter(y_train)
        count_test = collections.Counter(y_test)

        logging.info('Train: %s' % count_train)
        logging.info('Test: %s' % count_test)

        start_timeit = timeit.default_timer()

        classifier.best_estimator_.fit(x_train, y_train)
        y_pred_proba = classifier.best_estimator_.predict_proba(x_test)
        end_timeit = timeit.default_timer() - start_timeit

        n_test, n_labels = y_pred_proba.shape

        predicts = [Predict(n_test, dataset.levels, dataset.image.patch, 'max', y_pred_proba, y_test),
                    Predict(n_test, dataset.levels, dataset.image.patch, 'mult', y_pred_proba, y_test),
                    Predict(n_test, dataset.levels, dataset.image.patch, 'sum', y_pred_proba, y_test)]

        self.result = Result(count_train, count_test, dataset.image.patch, predicts, end_timeit)
