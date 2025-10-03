import collections
import itertools
import logging
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from classifier import get_classifier
from dataset import Dataset
from fold import Fold
from mean import Mean
from result import BestFold, BestMean
from save import SaveExperiment


class Experiment:
    def __init__(self, classifier: Any, data_augmentations: list, dataset: Dataset,
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
        self.data_augmentations = data_augmentations
        self.folds = folds
        self.indexes = None
        self.means = None
        self.metrics = metrics
        self.n_jobs = n_jobs
        self.rules = ["sum", "max", "mult"]
        self.save = None
        self.seed = seed
        self.verbose = verbose
        self.classifier = get_classifier(classifier, self.n_jobs, self.seed, self.verbose)

    def split_folds(self):
        x = np.random.rand(self.dataset.n_samples, self.dataset.n_features)
        y = [np.repeat(k, int(v / self.dataset.patch)) for k, v in dict(collections.Counter(self.dataset.y)).items()]
        y = np.array(list(itertools.chain(*y)))
        logging.info("StratifiedKFold x.shape: %s" % str(x.shape))
        logging.info("StratifiedKFold y.shape: %s" % str(y.shape))
        kf = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.seed)
        self.indexes = list(kf.split(x, y))

    def run(self, output):
        self.split_folds()
        folds = [Fold(self.dataset, fold, idx[0], idx[1]) for fold, idx in enumerate(self.indexes, start=1)]
        kwargs = {"cv": self.folds, "scoring": self.cv_metric, "n_jobs": self.n_jobs, "verbose": self.verbose}
        for fold in folds:
            fold.run(self.backend, self.classifier, self.data_augmentations, **kwargs)

        self.best_fold = BestFold(folds)

        self.means = [Mean(folds, rule) for rule in self.rules]
        self.best_mean = BestMean(self.means)

        self.save = SaveExperiment(self, folds, output)