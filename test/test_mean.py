from unittest import TestCase

import numpy as np
import pandas as pd

from fold import Fold
from level import Level
from mean import Mean


class TestMean(TestCase):
    n_levels = 23

    def best_eval(self, rules):
        data = {
            'accuracy': [0.5, 0.4, 0.3],
            'f1': [0.5, 0.4, 0.3],
            'rule': rules
        }
        return pd.DataFrame(data, index=None, columns=list(data.keys()))

    def create_top(self):
        data = {
            'k': np.repeat(np.arange(3, self.n_levels), 3),
            # "k";"topk_accuracy_score";"rule";"count_test";"topk_accuracy_score+100"
            'topk_accuracy_score': np.repeat(250, (self.n_levels - 3) * 3),
            'rule': (self.n_levels - 3) * ['sum', 'mult', 'max'],
            'count_test': np.repeat(500, (self.n_levels - 3) * 3),
            'topk_accuracy_score+100': np.repeat(0.5, (self.n_levels - 3) * 3),
        }
        return pd.DataFrame(data, index=None, columns=list(data.keys()))

    def create_eval(self, rules: list):
        folds = [Fold(None, None, None, None), Fold(None, None, None, None)]
        for f in folds:
            f.dataframes = {'evaluations': self.best_eval(rules), 'topk': self.create_top()}
        return folds

    def test_mean_max(self):
        folds = self.create_eval(['max', 'mult', 'sum'])
        mean = Mean(folds)
        self.assertEqual(mean.c.loc[mean.c['rule'] == 'max']['mean_accuracy'].values[0], 0.5)
        self.assertEqual(mean.c.loc[mean.c['rule'] == 'max']['mean_f1'].values[0], 0.5)

    def test_mean_mult(self):
        folds = self.create_eval(['mult', 'max', 'sum'])
        mean = Mean(folds)
        self.assertEqual(mean.c.loc[mean.c['rule'] == 'mult']['mean_accuracy'].values[0], 0.5)
        self.assertEqual(mean.c.loc[mean.c['rule'] == 'mult']['mean_f1'].values[0], 0.5)

    def test_mean_sum(self):
        folds = self.create_eval(['sum', 'mult', 'max'])
        mean = Mean(folds)
        self.assertEqual(mean.c.loc[mean.c['rule'] == 'sum']['mean_accuracy'].values[0], 0.5)
        self.assertEqual(mean.c.loc[mean.c['rule'] == 'sum']['mean_f1'].values[0], 0.5)

    def test_mean_top(self):
        folds = self.create_eval(['sum', 'mult', 'max'])
        mean = Mean(folds)
        self.assertTrue((mean.mean_top['mean_topk_accuracy_score'] == 250).all())
        self.assertTrue((mean.mean_top['mean_topk_accuracy_score+100'] == 0.5).all())
        self.assertTrue((mean.mean_top['mean_count_test'] == 500).all())