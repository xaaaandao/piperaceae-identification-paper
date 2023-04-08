import unittest

import numpy as np

from a import split_dataset, sum_rule, mult_rule, max_rule


class TestCython(unittest.TestCase):
    n_samples = 375
    n_features = 512
    n_labels = 5
    patch = 3
    x = np.arange(n_samples * patch * n_features).astype(np.float64)
    y = np.repeat(np.arange(5), int(n_samples / n_labels) * patch).astype(np.int16)

    def setUp(self) -> None:
        self.x = self.x.reshape((self.n_samples * self.patch, self.n_features))
        super().setUp()

    def test_split_dataset(self):
        index = [1, 4]

        start = (index[0] * self.patch) * self.n_features
        first_sample = np.arange(start, start + (self.n_features * self.patch))
        first_sample = first_sample.reshape((self.patch, self.n_features))

        start = (index[1] * self.patch) * self.n_features
        second_sample = np.arange(start, start + (self.n_features * self.patch))
        second_sample = second_sample.reshape((self.patch, self.n_features))

        xx = np.vstack((first_sample, second_sample))

        new_x, new_y = split_dataset(np.array(index), self.n_features, self.patch, self.x, self.y)
        self.assertEqual(True, np.array_equal(new_x, xx))

    def test_sum_rule(self):
        list_equal = []
        for i in range(10):
            y_pred_proba = np.random.random_sample((self.patch, self.n_labels))
            my_y_pred, my_y_score = sum_rule(3, self.n_labels, self.patch, y_pred_proba)
            y_pred_proba = np.sum(y_pred_proba, axis=0)
            y_pred = np.array([np.argmax(y_pred_proba)+1])
            list_equal.append(np.array_equal(my_y_pred, y_pred))

        self.assertEqual(True, all(equal for equal in list_equal))

    def test_mult_rule(self):
        list_equal = []
        for i in range(10):
            y_pred_proba = np.random.random_sample((self.patch, self.n_labels))
            my_y_pred, my_y_score = mult_rule(3, self.n_labels, self.patch, y_pred_proba)
            y_pred_proba = np.prod(y_pred_proba, axis=0)
            # +1 because not exists label 0
            y_pred = np.array([np.argmax(y_pred_proba)+1])
            list_equal.append(np.array_equal(my_y_pred, y_pred))

        self.assertEqual(True, all(equal for equal in list_equal))

    def test_max_rule(self):
        y_pred_proba = np.arange(69, dtype=np.float64).reshape(3, 23)
        my_y_pred, my_y_pred_score = max_rule(3, 23, 3, y_pred_proba)
        y_pred = [23]
        y_pred_score = np.array([y_pred_proba[2]])
        self.assertEqual(True, np.array_equal(my_y_pred, y_pred))
        self.assertEqual(True, np.array_equal(my_y_pred_score, y_pred_score))

if __name__ == '__main__':
    unittest.main()
