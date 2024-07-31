import os
from unittest import TestCase

import numpy as np
import pandas as pd

from dataset import Dataset
from fold import Fold
from image import Image
from level import Level
from mean import Mean
from result import Result


class TestMean(TestCase):
    rules = ['sum', 'max', 'mult']

    def create_result(self, rule):
        count_samples_patch = self.dataset.count_samples * self.dataset.image.patch
        self.x = np.random.uniform(0, 1, size=(count_samples_patch, self.dataset.count_features))
        self.y = np.random.randint(0, 1, size=(count_samples_patch, self.dataset.count_features), dtype=np.int16)
        self.first_fold = Fold(1, None, self.x, self.y)
        self.second_fold = Fold(2, None, self.x, self.y)
        self.first_fold.dataframes = {
            'evals': self.create_evals(rule),
            'tops': self.create_tops(rule),
            'true_positive': self.create_tps(rule),
        }
        self.second_fold.dataframes = {
            'evals': self.create_evals(rule),
            'tops': self.create_tops(rule),
            'true_positive': self.create_tps(rule),
        }
        return [self.first_fold, self.second_fold]

    def set_best_value(self, rule):
        values = np.random.uniform(0, 1, size=(len(self.rules),))
        idx = self.rules.index(rule)
        values[idx] = 1
        return values

    def create_evals(self, rule):
        data = {
            'accuracy': self.set_best_value(rule),
            'f1': self.set_best_value(rule),
            'rule': self.rules
        }
        return pd.DataFrame(data, index=None, columns=list(data.keys()))

    def create_tops(self, rule):
        count_k = (len(self.dataset.levels) - 3) * len(self.rules)
        data = {'k': np.repeat(np.arange(3, len(self.dataset.levels)), len(self.rules)),
                'topk_accuracy_score': np.random.uniform(0, 1, size=count_k),
                'count_test': np.repeat(1, count_k),
                'rule': (len(self.dataset.levels) - 3) * self.rules
                }
        df = pd.DataFrame(data, index=None, columns=list(data.keys()))
        df.loc[df['rule'] == rule, 'topk_accuracy_score'] = 1
        return df

    def create_tps(self, rule):
        labels = [l.specific_epithet for l in self.dataset.levels]
        data = {'labels': labels * len(self.rules),
                'rule': self.rules * len(labels),
                'true_positive': np.arange(1, len(labels) * len(self.rules) + 1),
                'count_train': np.arange(1, len(labels) * len(self.rules) + 1),
                'count_test': np.arange(1, len(labels) * len(self.rules) + 1)
                }
        df = pd.DataFrame(data, index=None, columns=list(data.keys()))
        df.loc[df['rule'] == rule, 'true_positive'] = 1
        df.loc[df['rule'] == rule, 'count_train'] = 1
        df.loc[df['rule'] == rule, 'count_test'] = 1
        return df

    def setUp(self):
        super().setUp()
        self.dataset = Dataset('./files/')
        self.dataset.load()

    def test_mean_sum(self):
        folds = self.create_result('sum')
        means = Mean(folds)
        df = means.dataframes['means']
        self.assertEqual(df.loc[df['rule'] == 'sum']['mean_accuracy'].values[0], 1)
        self.assertEqual(df.loc[df['rule'] == 'sum']['mean_f1'].values[0], 1)
        self.assertEqual(df.loc[df['rule'] == 'sum']['std_accuracy'].values[0], 0)
        self.assertEqual(df.loc[df['rule'] == 'sum']['std_f1'].values[0], 0)

    def test_mean_max(self):
        folds = self.create_result('max')
        means = Mean(folds)
        df = means.dataframes['means']
        self.assertEqual(df.loc[df['rule'] == 'max']['mean_accuracy'].values[0], 1)
        self.assertEqual(df.loc[df['rule'] == 'max']['mean_f1'].values[0], 1)
        self.assertEqual(df.loc[df['rule'] == 'max']['std_accuracy'].values[0], 0)
        self.assertEqual(df.loc[df['rule'] == 'max']['std_f1'].values[0], 0)

    def test_mean_mult(self):
        folds = self.create_result('mult')
        means = Mean(folds)
        df = means.dataframes['means']
        self.assertEqual(df.loc[df['rule'] == 'mult']['mean_accuracy'].values[0], 1)
        self.assertEqual(df.loc[df['rule'] == 'mult']['mean_f1'].values[0], 1)
        self.assertEqual(df.loc[df['rule'] == 'mult']['std_accuracy'].values[0], 0)
        self.assertEqual(df.loc[df['rule'] == 'mult']['std_f1'].values[0], 0)

    def test_tops_sum(self):
        folds = self.create_result('sum')
        means = Mean(folds)
        df = means.dataframes['tops']
        self.assertEqual(df.loc[df['rule'] == 'sum', 'mean_topk_accuracy_score'].all(), 1)
        self.assertEqual(df.loc[df['rule'] == 'sum', 'mean_count_test'].all(), 1)
        self.assertEqual(df.loc[df['rule'] == 'sum', 'std_topk_accuracy_score'].all(), 0)
        self.assertEqual(df.loc[df['rule'] == 'sum', 'std_count_test'].all(), 0)

    def test_tops_mult(self):
        folds = self.create_result('mult')
        means = Mean(folds)
        df = means.dataframes['tops']
        self.assertEqual(df.loc[df['rule'] == 'mult', 'mean_topk_accuracy_score'].all(), 1)
        self.assertEqual(df.loc[df['rule'] == 'mult', 'mean_count_test'].all(), 1)
        self.assertEqual(df.loc[df['rule'] == 'mult', 'std_count_test'].all(), 0)
        self.assertEqual(df.loc[df['rule'] == 'mult', 'std_count_test'].all(), 0)

    def test_tops_max(self):
        folds = self.create_result('max')
        means = Mean(folds)
        df = means.dataframes['tops']
        self.assertEqual(df.loc[df['rule'] == 'max', 'mean_topk_accuracy_score'].all(), 1)
        self.assertEqual(df.loc[df['rule'] == 'max', 'mean_count_test'].all(), 1)
        self.assertEqual(df.loc[df['rule'] == 'max', 'std_topk_accuracy_score'].all(), 0)
        self.assertEqual(df.loc[df['rule'] == 'max', 'std_count_test'].all(), 0)

    def test_tps_sum(self):
        folds = self.create_result('sum')
        means = Mean(folds)
        df = means.dataframes['true_positive']
        self.assertEqual(df.loc[df['rule'] == 'sum', 'mean_true_positive'].all(), 1)
        self.assertEqual(df.loc[df['rule'] == 'sum', 'mean_count_test'].all(), 1)
        self.assertEqual(df.loc[df['rule'] == 'sum', 'mean_count_train'].all(), 1)
        self.assertEqual(df.loc[df['rule'] == 'sum', 'std_true_positive'].all(), 0)
        self.assertEqual(df.loc[df['rule'] == 'sum', 'std_count_test'].all(), 0)
        self.assertEqual(df.loc[df['rule'] == 'sum', 'std_count_train'].all(), 0)

    def test_tps_max(self):
        folds = self.create_result('max')
        means = Mean(folds)
        df = means.dataframes['true_positive']
        self.assertEqual(df.loc[df['rule'] == 'max', 'mean_true_positive'].all(), 1)
        self.assertEqual(df.loc[df['rule'] == 'max', 'mean_count_test'].all(), 1)
        self.assertEqual(df.loc[df['rule'] == 'max', 'mean_count_train'].all(), 1)
        self.assertEqual(df.loc[df['rule'] == 'max', 'std_true_positive'].all(), 0)
        self.assertEqual(df.loc[df['rule'] == 'max', 'std_count_test'].all(), 0)
        self.assertEqual(df.loc[df['rule'] == 'max', 'std_count_train'].all(), 0)

    def test_tps_max(self):
        folds = self.create_result('mult')
        means = Mean(folds)
        df = means.dataframes['true_positive']
        self.assertEqual(df.loc[df['rule'] == 'mult', 'mean_true_positive'].all(), 1)
        self.assertEqual(df.loc[df['rule'] == 'mult', 'mean_count_test'].all(), 1)
        self.assertEqual(df.loc[df['rule'] == 'mult', 'mean_count_train'].all(), 1)
        self.assertEqual(df.loc[df['rule'] == 'mult', 'std_true_positive'].all(), 0)
        self.assertEqual(df.loc[df['rule'] == 'mult', 'std_count_test'].all(), 0)
        self.assertEqual(df.loc[df['rule'] == 'mult', 'std_count_train'].all(), 0)

    def test_tps_sum(self):
        folds = self.create_result('sum')
        means = Mean(folds)
        df = means.dataframes['true_positive']
        self.assertEqual(df.loc[df['rule'] == 'sum', 'mean_true_positive'].all(), 1)
        self.assertEqual(df.loc[df['rule'] == 'sum', 'mean_count_test'].all(), 1)
        self.assertEqual(df.loc[df['rule'] == 'sum', 'mean_count_train'].all(), 1)
        self.assertEqual(df.loc[df['rule'] == 'sum', 'std_true_positive'].all(), 0)
        self.assertEqual(df.loc[df['rule'] == 'sum', 'std_count_test'].all(), 0)
        self.assertEqual(df.loc[df['rule'] == 'mult', 'std_count_train'].all(), 0)
