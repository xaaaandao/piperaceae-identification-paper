from unittest import TestCase

import os
import numpy as np
import pandas as pd

from v1.dataset import DataAugmentation


class TestDataAugmentation(TestCase):

    def setUp(self):
        self.create_fake_array()
        self.create_fake_df_samples()
        super().setUp()

    def tearDown(self):
        if os.path.exists("test.npy"):
            os.remove("test.npy")
        if os.path.exists("samples.csv"):
            os.remove("samples.csv")
        super().tearDown()

    def create_fake_array(self):
        self.fake_data_augmentation = np.array([
            [1, 2, 3, 4, 1, "arquivo_a"],
            [1, 2, 3, 4, 1, "arquivo_a"],
            [1, 2, 3, 4, 1, "arquivo_a"],
            [1, 2, 3, 4, 1, "arquivo_b"],
            [1, 2, 3, 4, 1, "arquivo_b"],
            [1, 2, 3, 4, 1, "arquivo_b"],
            [1, 2, 3, 4, 2, "arquivo_c"],
            [1, 2, 3, 4, 2, "arquivo_c"],
            [1, 2, 3, 4, 2, "arquivo_c"],
            [1, 2, 3, 4, 3, "arquivo_d"],
            [1, 2, 3, 4, 3, "arquivo_d"],
            [1, 2, 3, 4, 3, "arquivo_d"]
        ])
        np.save("test.npy", self.fake_data_augmentation)

    def create_fake_df_samples(self):
        data = {
            "fold": [1, 1, 2, 3],
            "filename": ["arquivo_a", "arquivo_b", "arquivo_c", "arquivo_d"],
            "specific_epithet": ["especie_a", "especie_b", "arquivo_c", "arquivo_d"],
        }
        df = pd.DataFrame(data)
        df.to_csv("samples.csv", sep=";", quoting=2, index=False, header=True, encoding="utf-8")

    def test_load_samples(self):
        self.data_augmentation = DataAugmentation(".")
        self.assertEqual(len(self.data_augmentation.samples), self.fake_data_augmentation.shape[0])

    def test_filter_data(self):
        self.data_augmentation = DataAugmentation(".", 2)
        labels = self.data_augmentation.data[:, -2]
        self.assertTrue(all(l == 1 for l in labels))