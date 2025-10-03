import itertools
import logging
import os
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from best import BestMean, BestFold
from mean import Mean
from save import save_csv_transpose
from classifier import get_classifier
from dataset import Dataset
from fold import Fold


class Experiment:
    def __init__(self, classifier: Any, dataset: Dataset,
                 backend: str = "loky",
                 cv_metric: str = "f1_weighted", folds: int = 5,
                 metrics: list[str] = None, n_jobs: int = -1,
                 seed: int = 1234, verbose: int = 42):
        if metrics is None:
            metrics = ["f1", "accuracy"]

        self.backend = backend
        self.best_fold = None
        self.best_mean = None
        self.cv_metric = cv_metric
        self.dataset = dataset
        # self.data_augmentations = data_augmentations
        self.folds = folds
        self.indexes = list
        self.means = list
        self.metrics = metrics
        self.n_jobs = n_jobs
        self.rules = ["sum", "max", "mult"]
        self.seed = seed
        self.verbose = verbose
        self.classifier = get_classifier(classifier, self.n_jobs, self.seed, self.verbose)

    def get_indexs(self):
        x = np.random.rand(self.dataset.qtd_samples_no_patch, self.dataset.qtd_features)
        y = [np.repeat(k, int(v / self.dataset.patch)) for k, v in self.dataset.qtd_samples_label.items()]
        y = np.array(list(itertools.chain(*y)))

        logging.info("StratifiedKFold x.shape: %s" % str(x.shape))
        logging.info("StratifiedKFold y.shape: %s" % str(y.shape))

        kf = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.seed)
        self.indexes = list(kf.split(x, y))

    def run(self, output):
        self.get_indexs()
        folds = [Fold(self.dataset, fold, idx) for fold, idx in enumerate(self.indexes, start=1)]

        kwargs = {"cv": self.folds, "scoring": self.cv_metric, "n_jobs": self.n_jobs, "verbose": self.verbose}

        for fold in folds:
            fold.run(self.backend, self.classifier, output, **kwargs)

        self.best_fold = BestFold(folds)
        self.means = [Mean(folds, self.dataset.levels, output, self.dataset.patch, rule) for rule in self.rules]
        self.best_mean = BestMean(self.means)

        self.save(output)

    def save(self, output):
        self.save_best(output)
        self.save_info(output)

    def save_best(self, output):
        output_dir = os.path.join(output, "best")
        os.makedirs(output_dir, exist_ok=True)

        self.best_fold.save(output_dir)
        self.best_mean.save(output_dir)

    def save_info(self, output):
        data = self.to_dict()
        filename = os.path.join(output, "experiment.csv")
        save_csv_transpose(data, filename, header=False, index=True)

    def to_dict(self):
        return {
            "clf": [self.classifier.__class__.__name__],
            "folds": [self.folds],
            "metric": [self.metrics],
            "model": [self.dataset.model],
            "n_jobs": [self.n_jobs],
            "patch": [self.dataset.patch],
            "qtd_features": [self.dataset.qtd_features],
            "qtd_samples": [self.dataset.qtd_samples],
            "qtd_samples_no_patch": [self.dataset.qtd_samples_no_patch],
            "seed": [self.seed],
            "scoring": [self.seed],
            "verbose": [self.verbose],
        }
