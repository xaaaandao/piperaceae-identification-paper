import math
import os
import random
import unittest

import numpy as np

from dataset import Dataset
from result import Predict
from test_datasetbase import TestDatasetBase


class TestPredict(TestDatasetBase):
    def setUp(self):
        os.makedirs(self.dir_tmp, exist_ok=True)
        self.create_fake_dataset()
        self.dataset = Dataset(self.dir_tmp)
        self.y_pred_proba = self.dataset.x
        self.y_test = np.random.randint(1, self.max_labels+1, size=(self.qtd_samples,))

    def tearDown(self):
        for f in self.files:
            os.remove(os.path.join(self.dir_tmp, f))
        os.removedirs(self.dir_tmp)

    def test_predict_sum(self):
        self.predict = Predict(self.patch, "sum", self.y_pred_proba, self.y_test)
        pred_test = np.argmax(self.y_pred_proba[0:self.patch].sum(axis=0)) + 1
        self.assertEqual(pred_test, self.predict.y_pred[0])
        self.assertEqual(self.predict.y_true.shape[0], self.dataset.qtd_samples_no_patch)

    def test_predict_mult(self):
        self.predict = Predict(self.patch, "mult", self.y_pred_proba, self.y_test)
        group = self.y_pred_proba[0:self.patch]
        pred_test = np.argmax(group.prod(axis=0)) + 1
        self.assertEqual(pred_test, self.predict.y_pred[0])
        self.assertEqual(self.predict.y_true.shape[0], self.dataset.qtd_samples_no_patch)

    def test_predict_max(self):
        self.predict = Predict(self.patch, "max", self.y_pred_proba, self.y_test)
        group = self.y_pred_proba[0:self.patch]
        max_idx = np.argmax(group)
        row_in_group, col = np.unravel_index(max_idx, group.shape)
        self.assertEqual(col + 1, self.predict.y_pred[0])
        self.assertEqual(self.predict.y_true.shape[0], self.dataset.qtd_samples_no_patch)


if __name__ == '__main__':
    unittest.main()
