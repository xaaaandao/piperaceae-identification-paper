import collections
import logging
import os
import pathlib

import numpy as np
import pandas as pd

from sample import Sample, Level


class Dataset:
    data: np.ndarray = np.array([])
    filenames = []
    levels: list[Level] = []
    max_label: int = None
    model: str = None
    patch: int = 1
    samples: list[Sample] = []
    qtd_features: int
    qtd_samples: int
    qtd_samples_label: dict
    qtd_samples_no_patch: int
    x = []
    y = []

    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.load()
        self.update()
        self.print()

    def load(self):
        self.load_csv()
        self.load_npy()

    def find_path_sample_csv(self):
        path = pathlib.Path(self.input_dir)

        while path.name != "/" and not os.path.exists(os.path.join(path, "samples.csv")):
            path = path.parent
            if path.name == "/":
                raise FileNotFoundError("samples.csv not found")

        return path

    def get_level_by_name(self, name):
        for level in self.levels:
            if level.name == name:
                return level
        return None

    def load_csv(self):
        self.load_dataset()
        df = self.load_samples_csv()
        self.load_levels(df)
        self.load_samples(df)

    def load_dataset(self):
        filename = os.path.join(self.input_dir, "dataset.csv")
        df = pd.read_csv(filename, sep=";", encoding="utf-8", index_col=0, header=None)
        df = df.transpose()
        self.model = df["model"].values[0]
        self.patch = int(float(df["patch"].values[0]))

    def load_samples_csv(self):
        path = self.find_path_sample_csv()
        filename = os.path.join(path, "samples.csv")
        df = pd.read_csv(filename, sep=";")
        return df

    def load_levels(self, df):
        if len(self.levels) == 0:
            df = df[["fold", "specific_epithet"]].drop_duplicates()
            self.levels = [Level(row["fold"], row["specific_epithet"]) for idx, row in df.iterrows()]

    def load_npy(self):
        data = [np.load(f) for f in pathlib.Path(self.input_dir).rglob("*.npy")]

        if len(data) == 0:
            raise FileNotFoundError("No features found in %s" % self.input_dir)

        self.data = np.vstack(data)

        self.split_features_label()

    def load_samples(self, df):
        if len(self.samples) == 0:
            self.samples = [Sample(row["filename"], self.get_level_by_name(row["specific_epithet"])) for idx, row in df.iterrows()]

    def print(self):
        logging.info("input_dir: %s" % self.input_dir)
        # logging.info("height: %s width: %s" % (self.height, self.width))
        logging.info("model: %s qtd_features: %d" % (self.model, self.qtd_features))
        logging.info("qtd_samples: %d qtd_samples_no_patch: %d " % (self.qtd_samples, self.qtd_samples_no_patch))

    def update(self):
        self.qtd_features = self.x.shape[1]
        self.qtd_samples = self.x.shape[0]
        self.qtd_samples_label = collections.Counter(self.y)
        self.qtd_samples_no_patch = self.x.shape[0] // self.patch
        self.max_label = max(self.y)

    def split_features_label(self):
        self.x = self.data[:, :-2]
        self.y = self.data[:, -2]
        self.filenames = self.data[:, -1]

        self.x = self.x.astype(float)
        self.y = self.y.astype(float).astype(np.int16)
        logging.info("x.shape: %s" % str(self.x.shape))
        logging.info("y.shape: %s" % str(self.y.shape))

class DataAugmentation(Dataset):
    def __init__(self, input_dir, min_data_aug=-1):
        super().__init__(input_dir)
        self.input_dir = input_dir
        self.min_data_aug = min_data_aug