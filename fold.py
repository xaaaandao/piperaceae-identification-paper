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
    def __init__(self, fold: int, idx: Any, x: np.ndarray, y: np.ndarray, result: Result = None):
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

        self.create_dfs(dataset)

    def create_dfs(self, dataset):
        self.dfs = {
            'info': self.create_df_info(),
            'count_train_test': self.create_df_count_train_test(dataset),
            'evaluations': self.create_df_evaluations(),
            'predicts': self.create_df_predicts(dataset.levels),
            'topk': self.create_df_topk(),
            'true_positive': self.create_df_true_positive(dataset),
        }
        self.dfs.update({'best': self.create_df_best(self.dfs['evaluations'])})

    def create_df_evaluations(self):
        data = {
            'accuracy': [p.accuracy for p in sorted(self.predicts, key=lambda x: x.rule)],
            'f1': [p.f1 for p in sorted(self.predicts, key=lambda x: x.rule)],
            'rule': [p.rule for p in sorted(self.predicts, key=lambda x: x.rule)]
        }
        return pd.DataFrame(data, index=None, columns=list(data.keys()))

    def create_df_count_train_test(self, dataset):
        data = {'labels': [], 'trains': [], 'tests': []}
        for train, test in zip(sorted(self.count_train.items()), sorted(self.count_test.items())):
            data['trains'].append(train[1] / dataset.image.patch)
            data['tests'].append(test[1] / dataset.image.patch)
            data['labels'].append(train[0])

        df = pd.DataFrame(data, index=None, columns=list(data.keys()))
        df['labels'] = df[['labels']].map(
            lambda row: list(filter(lambda x: x.label.__eq__(row), dataset.levels))[0].specific_epithet)
        return df

    def create_df_info(self):
        data = {
            'time': [self.final_time],
            'total_test': [self.total_test],
            'total_train': [self.total_train],
            'total_test_no_patch': [self.total_test_no_patch],
            'total_train_no_patch': [self.total_train_no_patch],
        }
        return pd.DataFrame(data, index=None, columns=list(data.keys()))
        # print(df)

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
        return df

    def create_df_a(self, predict):
        data = {
            'k': [topk.k for topk in sorted(predict.topk, key=lambda x: x.k)],
            'topk_accuracy_score': [topk.top_k_accuracy_score for topk in sorted(predict.topk, key=lambda x: x.k)],
            'count_test': np.repeat(self.total_test_no_patch, len(predict.topk)),
            'topk_accuracy_score+100': [topk.top_k_accuracy_score / self.total_test_no_patch for topk in
                                        sorted(predict.topk, key=lambda x: x.k)],
            'rule': [predict.rule] * len(predict.topk)  # equivalent a np.repeat, but works in List[str]
        }
        return pd.DataFrame(data, columns=list(data.keys()))

    def create_df_topk(self):
        data = {'k': [], 'topk_accuracy_score': [], 'rule': []}
        df = pd.DataFrame(data, columns=list(data.keys()))
        for predict in self.predicts:
            ddf = self.create_df_a(predict)
            df = pd.concat([df, ddf], axis=0)
        return df

    def create_df_true_positive(self, dataset):
        data = {
            'labels': [],
            'true_positive': [],
            'rule': []
        }
        df = pd.DataFrame(data, columns=list(data.keys()))
        for predict in self.predicts:
            ddf = self.create_df_b(dataset, predict)
            df = pd.concat([df, ddf], axis=0)
        return df

    def get_count(self, count: dict, label: int):
        """
        Encontra a quantidade de treinos de uma determinada classe.
        :param count: coleção com todas as quantidades de treinos.
        :param label: classe que deseja encontrar.
        :return: quantidade de treinos de uma determinada classe.
        """
        for count in list(count.items()):
            if label == count[0]:
                return count[1]

    def get_level(self, dataset):
        """
        Cria uma lista com os levels (classes), ou seja, nome e a quantidade de treinos e testes, que serão utilizados na matriz de confusão.
        :param levels: levels (classes) com nome das espécies utilizadas.
        :param patch: quantidade de divisões da imagem.
        :return: lista com o nome das classes e a quantidade de treinos de testes.
        """
        return [level.specific_epithet for level in sorted(dataset.levels, key=lambda x: x.label)]

    def get_count_train(self, dataset):
        return [int(self.get_count(self.count_train, level.label) / dataset.image.patch)
                for level in sorted(dataset.levels, key=lambda x: x.label)]

    def get_count_test(self, dataset):
        return [int(self.get_count(self.count_test, level.label) / dataset.image.patch)
                for level in sorted(dataset.levels, key=lambda x: x.label)]

    def create_df_b(self, dataset, predict):
        data = {
            'labels': self.get_level(dataset),
            'count_train': self.get_count_train(dataset),
            'count_test': self.get_count_test(dataset),
            'true_positive': list(np.diag(predict.confusion_matrix)),
            'rule': [predict.rule] * len(dataset.levels)
        }
        return pd.DataFrame(data, columns=list(data.keys()))

    def create_df_best(self, df_evaluations):
        df_accuracy = df_evaluations.loc[df_evaluations['accuracy'].idxmax()]
        df_f1 = df_evaluations.loc[df_evaluations['f1'].idxmax()]
        data = {'metric': ['accuracy', 'f1'],
                'value': [df_accuracy['accuracy'], df_f1['f1']],
                'rule': [df_accuracy['rule'], df_f1['rule']]}
        return pd.DataFrame(data, columns=list(data.keys()))
