import os

from experiment import Experiment
from test_datasetbase import TestDatasetBase
from dataset import Dataset


class TestExperiment(TestDatasetBase):
    clf = "DecisionTreeClassifier"

    def setUp(self):
        os.makedirs(self.dir_tmp, exist_ok=True)
        self.create_fake_dataset()
        self.dataset = Dataset(self.dir_tmp)
        self.experiment = Experiment(self.clf, self.dataset, self.dir_tmp, folds=self.folds)
        self.experiment.run()

    def tearDown(self):
        for f in self.files:
            os.remove(os.path.join(self.dir_tmp, f))
        os.removedirs(self.dir_tmp)

    def test_split_folds(self):
        self.assertEqual(len(self.experiment.indexes), self.experiment.folds)

