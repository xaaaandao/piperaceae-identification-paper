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

import df
from arrays import split_dataset
from dataset import Dataset
from result import Result


# from test.result import Result


class Fold:
    def __init__(self, fold: int, idx: Any, x: np.ndarray, y: np.ndarray):
        self.fold = fold
        if idx and len(idx) > 0:
            self.idx_train = idx[0]
            self.idx_test = idx[1]
        self.x = x
        self.y = y
        self.count_train = None
        self.count_test = None
        self.dataframes = None
        self.final_time = None
        self.total_test = None
        self.total_train = None
        self.total_test_no_patch = None
        self.total_train_no_patch = None
        self.predicts = None

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

    def results(self, dataset):
        self.dataframes = {
            'classification_report': df.classifications(self.predicts),
            'confusion_matrix': df.confusion_matrix(self.count_train, self.count_test, dataset, self.predicts),
            'confusion_matrix_normalized': df.confusion_matrix_normalized(self.count_train, self.count_test, dataset, self.predicts),
            'confusion_matrix_multilabel': df.confusion_matrix_multilabel(self.predicts),
            'count_train_test': df.count_train_test(self.count_train, self.count_test, dataset),
            'evals': df.evals(self.predicts),
            'infos': df.infos(self.final_time, self.total_test, self.total_train, self.total_test_no_patch, self.total_train_no_patch),
            'preds': df.preds(dataset.levels, self.predicts),
            'tops': df.tops(self.predicts, self.total_test_no_patch),
            'true_positive': df.true_positive(self.count_train, self.count_test, dataset, self.predicts)
        }
        self.dataframes.update({'best_evals': df.best_evals(self.dataframes['evals'])})

    def save(self, output):
        output = os.path.join(output, 'fold+%d' % self.fold)
        os.makedirs(output, exist_ok=True)
        for k, v in self.dataframes.items():
            if isinstance(v, pd.DataFrame):
                filename = os.path.join(output, '%s.csv' % k)
                v.to_csv(filename, index=False, header=True, sep=';', quoting=2, encoding='utf-8')
                logging.info('Saved %s' % filename)
            if isinstance(v, dict):
                self.save_rules(k, output, v)

    def save_rules(self, k, output, v):
        d = os.path.join(output, k)
        if 'normalized' in k:
            d = os.path.join(output, 'confusion_matrix', 'normalized')
        os.makedirs(d, exist_ok=True)
        for k2, v2 in v.items():
            filename = os.path.join(d, '%s+%s.csv' % (k, k2))
            if 'multilabel' in k:
                d = os.path.join(output, 'confusion_matrix', 'multilabel', k2[1])
                os.makedirs(d, exist_ok=True)
                filename = os.path.join(d, '%s+%s.csv' % (k, k2[0]))
            v2.to_csv(filename, index=True, header=True, sep=';', quoting=2, encoding='utf-8')