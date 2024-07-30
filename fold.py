import collections
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
from result import Result


# from test.result import Result


class Fold:
    def __init__(self, fold: int, idx: Any, x: np.ndarray, y: np.ndarray, result: Result=None):
        self.fold = fold
        self.idx_train = idx[0]
        self.idx_test = idx[1]
        self.result = result
        self.x = x
        self.y = y

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

        # print(len(predicts))
        self.create_df(dataset)
        # predicts = [Predict(n_test, dataset.levels, dataset.image.patch, 'max', y_pred_proba, y_test),
        #             Predict(n_test, dataset.levels, dataset.image.patch, 'mult', y_pred_proba, y_test),
        #             Predict(n_test, dataset.levels, dataset.image.patch, 'sum', y_pred_proba, y_test)]

        # self.result = Result(self.count_train, self.count_test, dataset.image.patch, predicts, end_timeit)

    def create_df(self, dataset):
        self.create_df_info(dataset.image.patch)
        self.create_df_count_train_test(dataset)
        self.create_df_evaluations()
        self.create_df_predicts(dataset.levels)
        self.create_df_tok(dataset.image.patch)
        # self.create_df_true_positive()

    def create_df_evaluations(self):
        data = {
            'accuracy': [p.accuracy for p in sorted(self.predicts, key=lambda x: x.rule)],
            'f1': [p.f1 for p in sorted(self.predicts, key=lambda x: x.rule)],
            'rule': [p.rule for p in sorted(self.predicts, key=lambda x: x.rule)]
        }
        df = pd.DataFrame(data, index=None, columns=list(data.keys()))

    def create_df_count_train_test(self, dataset):
        data = { 'labels': [], 'trains': [], 'tests': [] }
        for train, test in zip(sorted(self.count_train.items()), sorted(self.count_test.items())):
            data['trains'].append(train[1]/dataset.image.patch)
            data['tests'].append(test[1]/dataset.image.patch)
            data['labels'].append(train[0])

        df = pd.DataFrame(data, index=None, columns=list(data.keys()))
        df['labels'] = df[['labels']].map(lambda row: list(filter(lambda x: x.label.__eq__(row), dataset.levels))[0].specific_epithet)

    def create_df_info(self, patch):
        data = {
            'time': [self.final_time],
            'total_test': [np.sum(list(self.count_test.values()))],
            'total_train': [np.sum(list(self.count_train.values()))],
            'total_test_no_patch': [[np.sum(list(self.count_test.values()))][0]/patch],
            'total_train_no_patch': [[np.sum(list(self.count_train.values()))][0]/patch],
        }
        df = pd.DataFrame(data, index=None, columns=list(data.keys()))
        print(df)

    def create_df_predicts(self, levels):
        data = {
            'y_pred+sum': list(itertools.chain(*[p.y_pred.tolist() for p in self.predicts if p.rule.__eq__('sum')])),
            'y_pred+mult': list(itertools.chain(*[p.y_pred.tolist() for p in self.predicts if p.rule.__eq__('mult')])),
            'y_pred+max': list(itertools.chain(*[p.y_pred.tolist() for p in self.predicts if p.rule.__eq__('max')])),
            'y_true': list(itertools.chain(*[self.predicts[0].y_true.tolist()]))
        }
        df = pd.DataFrame(data, index=None, columns=list(data.keys()))

        if len(levels) > 0:
            df = df.map(lambda row: list(filter(lambda x: x.label.__eq__(row), levels))[0].specific_epithet)

        df['equals'] = df.apply(lambda row: row[row == row['y_true']].index.tolist(), axis=1)

    def create_df_a(self, patch, predict):
        data = {
            'k': [topk.k for topk in sorted(predict.topk, key=lambda x: x.k)],
            'topk_accuracy_score': [topk.top_k_accuracy_score for topk in sorted(predict.topk, key=lambda x: x.k)],
            'count_test': np.repeat([np.sum(list(self.count_test.values()))][0]/patch, len(predict.topk)),
            'topk_accuracy_score+100': [topk.top_k_accuracy_score / ([np.sum(list(self.count_test.values()))][0]/patch) for topk in sorted(predict.topk, key=lambda x: x.k)],
            'rule': [predict.rule] * len(predict.topk)  # equivalent a np.repeat, but works in List[str]
        }
        return pd.DataFrame(data, columns=list(data.keys()))

    def create_df_tok(self, patch):
        data = {'k': [], 'topk_accuracy_score': [], 'rule': []}
        df = pd.DataFrame(data, columns=list(data.keys()))
        for predict in self.predicts:
            ddf = self.create_df_a(patch, predict)
            df = pd.concat([df, ddf], axis=0)
        print(df)

    def create_df_true_positive(self):
        pass