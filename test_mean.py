from unittest import TestCase

import numpy as np
import pandas as pd

from fold import Fold
from level import Level


class TestMean(TestCase):
    n_levels = 235

    def create_evaluations(self):
        data = {
            'accuracy': np.random.uniform(0, 1, size=(3, )),
            'f1': np.random.uniform(0, 1, size=(3, )),
            'rule': ['sum', 'mult', 'max']
        }
        return pd.DataFrame(data, index=None, columns=list(data.keys()))

    def create_top(self):
        data = {
            'k': [np.repeat(np.arange(3, self.n_levels), (self.n_levels - 4) * 3)],
            # -4 = -1 + -3
            # -1 => porque, k vai de 3 até 234 (ou seja, self.n_levels - 1)
            # -3 => porque, k começa em 3
            # * 3 => porque, tem as três regras (multiplicação, soma e maior)
            'topk_accuracy_score': np.random.uniform(0, 1, size=((self.n_levels - 4) * 3, )),
            'rule': [((self.n_levels - 4) * 3) * ['sum', 'mult', 'max']]
        }
        df = pd.DataFrame(data, columns=list(data.keys()))
        print(df)
        # for predict in self.predicts:
        #     ddf = self.create_df_a(predict)
        #     df = pd.concat([df, ddf], axis=0)
        # return df


    def setUp(self):
        super().setUp()
        # self.folds = self.create_levels()
        self.create_top()
        self.create_evaluations()

    def test_a(self):
        print('a')



