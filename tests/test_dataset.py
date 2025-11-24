import os
import unittest

from test_datasetbase import TestDatasetBase
from dataset import Dataset


class TestDataset(TestDatasetBase):

    def setUp(self):
        os.makedirs(self.dir_tmp, exist_ok=True)
        self.create_fake_dataset()
        self.dataset = Dataset(self.dir_tmp)

    def tearDown(self):
        for f in self.files:
            os.remove(os.path.join(self.dir_tmp, f))
        os.removedirs(self.dir_tmp)

    def test_levels(self):
        self.assertEqual(len(self.dataset.levels), self.max_labels)

    def test_samples(self):
        self.assertEqual(len(self.dataset.samples), self.qtd_samples//self.patch)

    def test_level_not_found(self):
        self.assertIsNone(self.dataset.get_level_by_name(self.create_fake_string()))

    def test_level_found(self):
        self.assertIsNotNone(self.dataset.get_level_by_name(self.fake_specific[0]))

    def test_patch(self):
        self.assertEqual(self.dataset.patch, self.patch)

    # def test_max_label(self):
    #     pass
        # self.assertEqual(self.dataset.max_label, self.max_level)

    def test_model_name(self):
        self.assertEqual(self.dataset.model, self.fake_model_name[0])


if __name__ == '__main__':
    unittest.main()
