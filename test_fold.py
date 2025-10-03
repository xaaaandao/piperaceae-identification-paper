import math
import os
import random
import unittest

import numpy as np
import numpy.testing as npt

from dataset import Dataset
from experiment import Experiment
from fold import Fold
from test_datasetbase import TestDatasetBase


class TestFold(TestDatasetBase):
    backend = "loky"
    clf = "DecisionTreeClassifier"
    cv_metric = "f1_weighted"
    n_jobs = -1
    verbose = 42

    def setUp(self):
        os.makedirs(self.dir_tmp, exist_ok=True)
        self.create_fake_dataset()
        self.dataset = Dataset(self.dir_tmp)
        self.create_fold()
        self.experiment = Experiment(self.clf, self.dataset, self.dir_tmp, folds=self.folds)
        self.experiment.get_indexs()
        self.folds = [Fold(self.dataset, f, self.experiment.indexes) for f in range(self.folds)]

    def tearDown(self):
        for f in self.files:
            os.remove(os.path.join(self.dir_tmp, f))
        os.removedirs(self.dir_tmp)

    def create_idxs(self):
        idxs = np.arange(0, self.dataset.qtd_samples_no_patch)
        self.idx_train = idxs[0: len(idxs) // 2]
        self.idx_test = idxs[len(idxs) // 2:]
        self.idxs = list((self.idx_train, self.idx_test))

    def create_fold(self):
        self.create_idxs()
        self.fold = Fold(self.dataset, self.patch, self.idxs)

    def test_idx(self):
        npt.assert_array_equal(self.fold.idx.idx_train, self.idx_train)
        npt.assert_array_equal(self.fold.idx.idx_test, self.idx_test)

    def test_split_folds(self):
        x_train, y_train, filenames_train = self.fold.split_fold(self.idx_train)
        npt.assert_array_equal(self.dataset.x[0:self.patch], x_train[0:self.patch])

    def test_split_folds_multiple_folds(self):
        for f in self.folds:
            x_train, y_train, filenames_train = f.split_fold(f.idx.idx_train[0])
            self.assertTrue(x_train.shape[0] % self.dataset.patch == 0)
            self.assertTrue(y_train.shape[0] % self.dataset.patch == 0)
            self.assertTrue(filenames_train.shape[0] % self.dataset.patch == 0)



if __name__ == '__main__':
    unittest.main()
