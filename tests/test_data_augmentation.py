import collections
from unittest import TestCase

import os
import numpy as np
import numpy.testing as npt
import pandas as pd

from dataset import Dataset, DataAugmentation
from experiment import Experiment
from fold import Fold, Data
from test_datasetbase import TestDatasetBase


class TestDataAugmentation(TestDatasetBase):
    clf = "DecisionTreeClassifier"

    def setUp(self):
        os.makedirs(self.dir_tmp, exist_ok=True)
        self.create_fake_dataset()
        self.dataset = Dataset(self.dir_tmp)
        self.data_augmentations = [DataAugmentation(self.dir_tmp)]
        self.create_fold()
        self.experiment = Experiment(self.clf, self.dataset, self.data_augmentations, self.dir_tmp, cv=self.folds)
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

    def test_merge_data_augmentation(self):
        for f in self.folds:
            x_train, y_train, filenames_train = f.split_fold(f.idx.idx_train[0])
            f.train = Data(x_train, y_train, filenames_train, self.dataset.patch)

            train_x_shape = f.train.x.shape[0]

            f.find_train_data_augmentation(self.data_augmentations)
            diff = np.setdiff1d(f.filenames_aug, f.train.filenames)
            diff2 = np.setdiff1d(f.train.filenames, f.filenames_aug)
            self.assertTrue(len(diff) == 0)
            self.assertTrue(len(diff2) == 0)

            f.merge_data_augmentation()
            self.assertEqual(train_x_shape + f.x_aug.shape[0], f.train.x.shape[0])











