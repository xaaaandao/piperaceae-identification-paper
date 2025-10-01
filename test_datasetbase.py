import itertools
import math
import os
import random
import string
import unittest

import numpy as np
import pandas as pd


class TestDatasetBase(unittest.TestCase):
    dir_tmp = "./tmp"
    files = ["dataset.csv", "samples.csv", "fake.npy"]
    max_tam_string = np.random.randint(1, 10)
    qtd_features = np.random.randint(1, 10)

    max_labels = np.random.randint(2, 10)
    folds, patch = random.sample(range(2, 11), 2)

    def create_fake_dataset(self):
        self.create_fake()
        self.create_fake_csv()

    def create_fake(self):
        self.create_fake_values()
        self.fake_model_name = self.create_fake_string()
        self.fake_filenames = self.create_fake_string(qtd=self.qtd_samples//self.patch)
        self.fake_specific = self.create_fake_string(qtd=self.max_labels)
        self.create_fake_npy()

    def create_fake_csv(self):
        self.create_fake_dataset_csv()
        self.create_fake_sample_csv()

    def create_fake_dataset_csv(self):
        data = {
            "model": self.fake_model_name,
            "patch": [self.patch],
        }
        filename = os.path.join(self.dir_tmp, "dataset.csv")
        df = pd.DataFrame(data, columns=list(data.keys()))
        df = df.transpose()
        df.to_csv(filename, sep=";", quoting=2, index=True, header=False, encoding="utf-8")

    def create_fake_npy(self):
        filename = os.path.join(self.dir_tmp, "fake.npy")
        fake_x = np.random.rand(self.qtd_samples, self.qtd_features)
        fake_y = self.create_fake_y()
        fake_filenames = self.fake_filenames * self.patch
        fake_data = np.column_stack([fake_x, fake_y, fake_filenames])
        np.save(filename, fake_data)

    def create_fake_data_sample_csv(self):
        specific_epithet = [np.repeat(self.fake_specific[k - 1], v // self.patch) for k, v in
                                self.qtd_samples_label.items()]
        specific_epithet = list(itertools.chain(*specific_epithet))
        fold = self.create_fake_y(self.patch)
        return fold, specific_epithet

    def create_fake_sample_csv(self):
        fold, specific_epithet = self.create_fake_data_sample_csv()
        data = {
            "filename": self.fake_filenames,
            "specific_epithet": specific_epithet,
            "fold": fold,
        }
        filename = os.path.join(self.dir_tmp, "samples.csv")
        df = pd.DataFrame(data)
        df.to_csv(filename, sep=";", quoting=2, index=False, header=True, encoding="utf-8")

    def create_fake_string(self, qtd=1):
        alphabets = np.array(list(string.ascii_letters))
        indices = np.random.randint(0, len(alphabets), size=(qtd, self.max_tam_string))
        return [''.join(alphabets[idx]) for idx in indices]

    def create_fake_values(self):
        self.set_mmc()
        self.qtd_samples_label = {i: random.randint(self.folds, 50) * self.mmc for i in range(1, self.max_labels + 1)}
        self.qtd_samples = np.sum([v for v in self.qtd_samples_label.values()])

    def create_fake_y(self, patch=1):
        fold = [np.repeat(k, v // patch) for k, v in self.qtd_samples_label.items()]
        return list(itertools.chain(*fold))

    def set_mmc(self):
        self.mmc = abs(self.folds * self.patch) // math.gcd(self.folds, self.patch)