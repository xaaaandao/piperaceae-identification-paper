import collections
import dataclasses
import logging
import os
import pathlib
from typing import Any

import numpy as np
import pandas as pd

from level import Level, get_level_by_name
from sample import Sample


@dataclasses.dataclass(init=False)
class Dataset:
    height: int
    filenames: list
    input_dir: str
    levels: list
    model: str
    n_features: int
    n_labels: int
    n_samples: int
    n_samples_patch: int
    patch: int
    samples: list
    width: int
    x: Any
    y: Any

    def __init__(self, input_dir):
        self.input_dir = input_dir
        if os.path.exists(os.path.join(self.input_dir, "dataset.csv")):
            self.load_csv()
        else:
            self.csv_file_is_empty()

        path = pathlib.Path(self.input_dir)

        while path.name != "/" and not os.path.exists(os.path.join(path, "samples.csv")):
            path = path.parent

        self.load_samples(path)

    def load_csv(self):
        filename = os.path.join(self.input_dir, "dataset.csv")
        df = pd.read_csv(filename, sep=";", encoding="utf-8", index_col=0, header=None)
        df = df.transpose()
        self.height = int(float(df["height"].values[0]))
        self.model = df["model"].values[0]
        self.n_features = int(float(df["n_features"].values[0]))
        self.n_labels = int(float(df["n_labels"].values[0]))
        self.n_samples = int(float(df["n_samples"].values[0]))
        self.n_samples_patch = int(float(df["n_samples+patch"].values[0]))
        self.patch = int(float(df["patch"].values[0]))
        self.width = int(float(df["width"].values[0]))

    def csv_file_is_empty(self):
        self.data_aug = None
        self.height = None
        self.model = None
        self.n_features = None
        self.n_samples = None
        self.n_samples_patch = None
        self.patch = None
        self.width = None

    def print(self):
        logging.info("input_dir: %s" % self.input_dir)
        logging.info("height: %s width: %s" % (self.height, self.width))
        logging.info("model: %s n_features: %s n_labels: %s" % (self.model, self.n_features, self.n_labels))
        logging.info("n_samples: %s n_samples+patch: %s patch: %s" % (self.n_samples, self.n_samples_patch, self.patch))

    def load_features(self):
        features = [np.load(p) for p in pathlib.Path(self.input_dir).rglob("*.npy")]

        if len(features) == 0:
            raise FileNotFoundError("no features found in %s" % self.input_dir)

        self.split_features_labels(features)

        if len(self.levels) == 0:
            self.levels = [Level(idx, "Espécie %d" % idx) for idx in range(np.min(self.y), np.max(self.y) + 1)]

    def split_features_labels(self, features: list[Any]):
        features = np.vstack(features)
        self.x = features[:, :-2]
        self.y = features[:, -2]
        self.filenames = features[:, -1]

        self.x = self.x.astype(float)
        self.y = self.y.astype(float).astype(np.int16)
        logging.info("x.shape: %s" % str(self.x.shape))
        logging.info("y.shape: %s" % str(self.y.shape))

    def load_samples(self, path):
        filename = os.path.join(path, "samples.csv")

        if not os.path.exists(filename):
            self.levels = []
            self.samples = []
        else:
            self.load_samples_csv(filename)

    def load_samples_csv(self, filename: str):
        # TODO considerar quando nao tem samples.csv
        df = pd.read_csv(filename, sep=";", encoding="utf-8", index_col=None)
        dfs = df[["fold", "specific_epithet"]].drop_duplicates()
        self.levels = [Level(row["fold"], row["specific_epithet"]) for idx, row in dfs.iterrows()]
        self.samples = [Sample(row["filename"], get_level_by_name(self.levels, row["specific_epithet"])) for idx, row in df.iterrows()]

class DataAugmentation(Dataset):
    def __init__(self, input_dir, min_class=-1):
        if input_dir is not None and os.path.exists(input_dir):
            super().__init__(input_dir)
            self.data = None
            self.min_class = min_class
            self.load_data_augmentation()

    def load_data_augmentation(self):
        data = [np.load(p) for p in pathlib.Path(self.input_dir).rglob("*.npy")]
        self.data = np.vstack(data)
        logging.info("x_augmented: %s" % str(self.data.shape))
        self.filter_data()

    def filter_data(self):
        if self.min_class > 0:
            min_labels = collections.Counter(s.level.label for s in self.samples)
            min_labels = [k for k, v in min_labels.items() if self.min_class <= v]

            labels = self.data[:, -2].astype(float).astype(np.int16)
            self.data = self.data.astype(object)
            self.data[:, -2] = labels

            self.data = self.data[np.isin(labels, min_labels)]
            logging.info("x_augmented FILTERED: %s" % str(self.data.shape))
