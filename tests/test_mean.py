import os
import unittest

import numpy as np

from dataset import Dataset
from experiment import Experiment
from fold import Fold
from mean import Mean
from predict import Predict
from result import Result
from test_datasetbase import TestDatasetBase


class TestMean(TestDatasetBase):
    backend = "loky"
    clf = "DecisionTreeClassifier"
    cv_metric = "f1_weighted"
    n_jobs = -1
    output = "./output"
    rules = ["sum", "mult", "max"]
    verbose = 42

    def setUp(self):
        os.makedirs(self.dir_tmp, exist_ok=True)
        self.create_fake_dataset()
        self.dataset = Dataset(self.dir_tmp)
        self.experiment = Experiment(self.clf, self.dataset, [], cv=self.folds)
        self.experiment.get_indexs()
        self.folds = [Fold(self.dataset, f, self.experiment.indexes) for f in range(self.folds)]
        self.create_predicts()
        self.create_results()
        self.create_means()

    def create_predicts(self):
        self.y_pred_proba = np.random.rand(self.dataset.x.shape[0], self.max_labels)
        self.y_test = np.random.randint(1, self.max_labels + 1, size=(self.qtd_samples,))
        self.predicts = [Predict(self.patch, r, self.y_pred_proba, self.y_test) for r in self.rules]

    def create_means(self):
        for f in self.folds:
            f.results = self.results
        self.means = [Mean(self.folds, self.dataset.levels, rule) for rule in self.rules]

    def create_results(self):
        self.results = [Result(self.dataset.levels, p) for p in self.predicts]

    def tearDown(self):
        for f in self.files:
            os.remove(os.path.join(self.dir_tmp, f))
        os.removedirs(self.dir_tmp)

    def test_mean(self):
        for m in self.means:
            if len(self.dataset.levels) > 3:
                self.assertEqual(len(m.tops), len(self.dataset.levels)-3)
            self.assertEqual(len(m.levels), self.max_labels)


if __name__ == '__main__':
    unittest.main()
