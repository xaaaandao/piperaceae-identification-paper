import math
import random
import unittest

import numpy as np

from arrays import max_rule, sum_rule, mult_rule


class TestPredict(unittest.TestCase):
    max_labels = np.random.randint(2, 10)
    folds, patch = random.sample(range(2, 11), 2)

    def setUp(self):
        self.create_fake_values()
        self.create_y_pred_proba()

    def create_fake_values(self):
        self.set_mmc()
        self.qtd_test = random.randint(self.folds, 50) * self.mmc

    def create_y_pred_proba(self):
        self.y_pred_proba = np.random.rand(self.qtd_test, self.max_labels)

    def set_mmc(self):
        self.mmc = abs(self.folds * self.patch) // math.gcd(self.folds, self.patch)

    def test_max_predict(self):
        self.y_test, self.y_score = max_rule(self.qtd_test, self.max_labels, self.patch, self.y_pred_proba)
        for y in self.y_test:
            for s in self.y_score:
                print(s, np.max(s), s[y - 1], y)
                self.assertEqual(s[y - 1], np.max(s))

    def test_sum_predict(self):
        self.y_test, self.y_score = sum_rule(self.qtd_test, self.max_labels, self.patch, self.y_pred_proba)
        for y in self.y_test:
            for s in self.y_score:
                print(s, np.max(s), s[y - 1], y)
                self.assertEqual(s[y - 1], np.max(s))

    def test_mult_predict(self):
        self.y_test, self.y_score = mult_rule(self.qtd_test, self.max_labels, self.patch, self.y_pred_proba)
        for y in self.y_test:
            for s in self.y_score:
                print(s, np.max(s), s[y - 1], y)
                self.assertEqual(s[y - 1], np.max(s))

if __name__ == '__main__':
    unittest.main()
