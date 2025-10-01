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

    def create_fake_values(self):
        self.set_mmc()
        self.qtd_samples_label = {i: random.randint(self.folds, 50) * self.mmc for i in range(1, self.max_labels + 1)}
        self.qtd_test = random.randint(self.folds, 10) * self.mmc
        self.y_pred_proba = np.random.rand(self.qtd_test, self.max_labels)

    def set_mmc(self):
        self.mmc = abs(self.folds * self.patch) // math.gcd(self.folds, self.patch)

    def test_max_predict(self):
        self.y_test, self.y_score = max_rule(self.qtd_test, self.max_labels, self.patch, self.y_pred_proba)
        for yt, ys in zip(self.y_test, self.y_score):
            print(yt, np.argmax(ys), ys)
            self.assertEqual(yt - 1, np.argmax(ys))

    def test_sum_predict(self):
        self.y_test, self.y_score = sum_rule(self.qtd_test, self.max_labels, self.patch, self.y_pred_proba)
        for yt, ys in zip(self.y_test, self.y_score):
            self.assertEqual(yt - 1, np.argmax(ys))


    def test_mult_predict(self):
        self.y_test, self.y_score = mult_rule(self.qtd_test, self.max_labels, self.patch, self.y_pred_proba)
        for yt, ys in zip(self.y_test, self.y_score):
            self.assertEqual(yt - 1, np.argmax(ys))


if __name__ == '__main__':
    unittest.main()
