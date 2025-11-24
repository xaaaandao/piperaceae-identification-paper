import collections
import itertools
import tempfile
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
    cut_mix = "./cut-mix"

    def setUp(self):
        os.makedirs(self.dir_tmp, exist_ok=True)
        os.makedirs(self.cut_mix, exist_ok=True)
        self.create_fake_dataset()
        self.dataset = Dataset(self.dir_tmp)

    def tearDown(self):
        for f in self.files:
            os.remove(os.path.join(self.dir_tmp, f))
        os.removedirs(self.dir_tmp)
        # os.removedirs(self.cut_mix)

    def create_idxs(self):
        idxs = np.arange(0, self.dataset.qtd_samples_no_patch)
        self.idx_train = idxs[0: len(idxs) // 2]
        self.idx_test = idxs[len(idxs) // 2:]
        self.idxs = list((self.idx_train, self.idx_test))

    def create_fold(self):
        self.create_idxs()
        self.fold = Fold(self.dataset, self.patch, self.idxs)

    def create_one_data_augmentation(self, min_data_augmentation=-1):
        self.data_augmentations = [DataAugmentation(self.dir_tmp, min_data_augmentation)]
        self.create_fold()
        self.experiment = Experiment(self.clf, self.dataset, self.data_augmentations, cv=self.folds)
        self.experiment.get_indexs()
        self.folds = [Fold(self.dataset, f, self.experiment.indexes) for f in range(self.folds)]

    def test_merge_one_data_augmentation(self):
        self.create_one_data_augmentation()
        for f in self.folds:
            x_train, y_train, filenames_train = f.split_fold(f.idx.idx_train[0])
            f.train = Data(x_train, y_train, filenames_train, self.dataset.patch)

            train_x_shape = f.train.x.shape[0]

            f.find_train_data_augmentation(self.data_augmentations)
            diff = np.setdiff1d(f.filenames_aug, f.train.filenames)
            diff2 = np.setdiff1d(f.train.filenames, f.filenames_aug)
            self.assertTrue(len(diff) == 0)
            self.assertTrue(len(diff2) == 0)

            f.merge_train_data_augmentation()
            self.assertEqual(train_x_shape + f.x_aug.shape[0], f.train.x.shape[0])

    def test_merge_min_data_augmentation(self):
        minimum_samples = np.min(list(collections.Counter(s.level.label for s in self.dataset.samples).values()))
        minimum_label = [k for k, v in collections.Counter(s.level.label for s in self.dataset.samples).items() if v <= minimum_samples+1]

        self.create_one_data_augmentation(min_data_augmentation=minimum_samples+1)
        labels_filtered = self.data_augmentations[0].data[:, -2]
        self.assertTrue(len(np.setdiff1d(labels_filtered, minimum_label)) == 0)

    def test_cut_mix(self):
        self.dir_tmp = "./cut-mix"
        self.create_fake_dataset()
        self.dataset = Dataset(self.dir_tmp)
        self.create_one_data_augmentation()
        for f in self.folds:
            x_train, y_train, filenames_train = f.split_fold(f.idx.idx_train[0])
            f.train = Data(x_train, y_train, filenames_train, self.dataset.patch)

            train_x_shape = f.train.x.shape[0]

            f.find_train_data_augmentation(self.data_augmentations)
            filenames = ["+".join(c) for c in itertools.combinations(f.train.filenames, 2)]
            '''
                Se todos os elementos de f.filenames_aug estão em filenames
                significa que eu estou usando todas amostras.
                Se fosse o contrário o resultado é maior que zero, porque filenames
                não gera as combinações por label.
            '''
            diff = np.setdiff1d(f.filenames_aug, filenames)
            self.assertTrue(len(diff) == 0)

            f.merge_train_data_augmentation()
            self.assertEqual(train_x_shape + f.x_aug.shape[0], f.train.x.shape[0])
