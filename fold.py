import collections
import dataclasses
import logging
import os
import pathlib
import timeit
from typing import Any, LiteralString

import numpy as np
import pandas as pd

from arrays import split_dataset
from dataset import Dataset
from predict import Predict
from result import Result


class Fold:
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

        self.count_train = collections.Counter(y_train)
        self.count_test = collections.Counter(y_test)

        logging.info('Train: %s' % self.count_train)
        logging.info('Test: %s' % self.count_test)

        start_timeit = timeit.default_timer()

        classifier.best_estimator_.fit(x_train, y_train)
        y_pred_proba = classifier.best_estimator_.predict_proba(x_test)
        end_timeit = timeit.default_timer() - start_timeit

        n_test, n_labels = y_pred_proba.shape

        predicts = [Predict(n_test, dataset.levels, dataset.image.patch, 'max', y_pred_proba, y_test),
                    Predict(n_test, dataset.levels, dataset.image.patch, 'mult', y_pred_proba, y_test),
                    Predict(n_test, dataset.levels, dataset.image.patch, 'sum', y_pred_proba, y_test)]

        self.result = Result(self.count_train, self.count_test, dataset.image.patch, predicts, end_timeit)

    def save_count_train_test(self, output: pathlib.Path | LiteralString | str):
        filename = os.path.join(output, 'count_train_test.csv')

        logging.info('Saving')
        data = { 'labels': [], 'trains': [], 'tests': [] }
        for train, test in zip(sorted(self.count_train.items()), sorted(self.count_test.items())):
            data['trains'].append(train[1])
            data['tests'].append(test[1])
            data['labels'].append(train[0])

        df = pd.DataFrame(data, columns=data.keys())
        # TODO save_csv

    def save(self, levels:list, output: pathlib.Path | LiteralString | str):
        output = os.path.join(output, 'fold+%d' % (self.fold))
        os.makedirs(output, exist_ok=True)
        self.save_count_train_test(output)
        self.result.save(levels, output)