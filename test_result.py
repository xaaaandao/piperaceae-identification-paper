import os
import shutil
import unittest

import numpy as np

from dataset import Dataset
from experiment import Experiment
from fold import Fold
from predict import Predict
from result import Result
from test_datasetbase import TestDatasetBase


class TestResult(TestDatasetBase):
    backend = "loky"
    clf = "DecisionTreeClassifier"
    cv_metric = "f1_weighted"
    n_jobs = -1
    output = "./output"
    verbose = 42

    def setUp(self):
        os.makedirs(self.dir_tmp, exist_ok=True)
        self.create_fake_dataset()
        self.dataset = Dataset(self.dir_tmp)
        self.experiment = Experiment(self.clf, self.dataset, folds=self.folds)
        self.experiment.get_indexs()
        self.folds = [Fold(self.dataset, f, self.experiment.indexes) for f in range(self.folds)]
        self.create_predict()
        self.result = Result(self.folds[0].fold, self.dataset.levels, self.output, self.predict)

    def tearDown(self):
        for f in self.files:
            os.remove(os.path.join(self.dir_tmp, f))
        os.removedirs(self.dir_tmp)
        shutil.rmtree(self.output)

    def create_predict(self):
        self.y_pred_proba = np.random.rand(self.dataset.x.shape[0], self.max_labels)
        self.y_test = np.random.randint(1, self.max_labels + 1, size=(self.qtd_samples,))
        self.predict = Predict(self.patch, "sum", self.y_pred_proba, self.y_test)

    def test_tops(self):
        if len(self.dataset.levels) > 3:
            self.assertEqual(len(self.dataset.levels)-3, len(self.result.tops))
        else:
            self.assertTrue(len(self.result.tops) == 0)

if __name__ == '__main__':
    unittest.main()
