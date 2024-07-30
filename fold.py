import collections
import csv
import itertools
import logging
import os
import pathlib
import timeit
from typing import Any, LiteralString

import numpy as np
import pandas as pd

from arrays import split_dataset
from dataset import Dataset
from df import DF
from result import Result


# from test.result import Result


class Fold:
    def __init__(self, fold: int, idx: Any, x: np.ndarray, y: np.ndarray,
                 count_train: Any = None,
                 count_test: Any = None,
                 dataframes: Any = None,
                 final_time: float = None,
                 total_test: int = None,
                 total_train: int = None,
                 total_test_no_patch: int = None,
                 total_train_no_patch: int = None,
                 predicts: dict = None,
                 result: Result = None):
        self.fold = fold
        if idx and len(idx) > 0:
            self.idx_train = idx[0]
            self.idx_test = idx[1]
        self.x = x
        self.y = y
        self.count_train = count_train
        self.count_test = count_test
        self.dfs = None
        self.dataframes = dataframes
        self.final_time = final_time
        self.total_test = total_test
        self.total_train = total_train
        self.total_test_no_patch = total_test_no_patch
        self.total_train_no_patch = total_train_no_patch
        self.predicts = predicts
        self.result = result

    def run(self, classifier: Any, dataset: Dataset):
        """
        Separa o dataset (treino e teste), usa o classificador para treinar e predizer.
        Com a predição é aplicado a regra da soma, multiplicação e máximo.
        Por fim, é feito gerado as métricas.
        :param classifier: classificador com os melhores hiperparâmetros.
        :param dataset: classe com as informações do dataset.
        """
        x_train, y_train = split_dataset(self.idx_train, dataset.count_features, dataset.image.patch, self.x, self.y)
        x_test, y_test = split_dataset(self.idx_test, dataset.count_features, dataset.image.patch, self.x, self.y)

        self.count_train = collections.Counter(y_train)
        self.count_test = collections.Counter(y_test)
        self.total_test = np.sum(list(self.count_test.values()))
        self.total_train = np.sum(list(self.count_train.values()))
        self.total_test_no_patch = self.total_test / dataset.image.patch
        self.total_train_no_patch = self.total_train / dataset.image.patch

        logging.info('Train: %s' % self.count_train)
        logging.info('Test: %s' % self.count_test)

        start_timeit = timeit.default_timer()

        classifier.best_estimator_.fit(x_train, y_train)
        y_pred_proba = classifier.best_estimator_.predict_proba(x_test)
        self.final_time = timeit.default_timer() - start_timeit

        n_test, n_labels = y_pred_proba.shape

        self.predicts = [Result(n_test, dataset.levels, dataset.image.patch, 'max', y_pred_proba, y_test),
                         Result(n_test, dataset.levels, dataset.image.patch, 'mult', y_pred_proba, y_test),
                         Result(n_test, dataset.levels, dataset.image.patch, 'sum', y_pred_proba, y_test)]

        self.dfs = DF(self.count_train,
                      self.count_test,
                      dataset,
                      self.final_time,
                      self.predicts,
                      self.total_train,
                      self.total_test,
                      self.total_train_no_patch,
                      self.total_test_no_patch)

    def save(self, output):
        output = os.path.join(output, 'fold+%d' % self.fold)
        os.makedirs(output, exist_ok=True)
        self.dfs.save(output)
