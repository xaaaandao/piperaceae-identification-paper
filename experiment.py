import itertools
import logging
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from result import Mean
from v1.classifier import get_classifier
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
        # self.best_mean = None
        self.cv_metric = cv_metric
        self.dataset = dataset
        # self.data_augmentations = data_augmentations
        self.folds = folds
        self.indexes = list
        # self.means = None
        self.metrics = metrics
        self.n_jobs = n_jobs
        self.rules = ["sum", "max", "mult"]
        # self.save = None
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

    def run(self):
        self.get_indexs()
        folds = [Fold(self.dataset, fold, idx) for fold, idx in enumerate(self.indexes, start=1)]

        kwargs = {"cv": self.folds, "scoring": self.cv_metric, "n_jobs": self.n_jobs, "verbose": self.verbose}

        for fold in folds:
            fold.run(self.backend, self.classifier, **kwargs)

        self.best_fold = BestFold(folds)
        #
        self.means = [Mean(folds, rule) for rule in self.rules]
        # self.best_mean = BestMean(self.means)
        #
        # self.save = SaveExperiment(self, folds, output)

class BestFold:
    def __init__(self, folds):
        self.f1 = max(folds, key=lambda x: x.best_result.f1.f1)
        self.accuracy = max(folds, key=lambda x: x.best_result.accuracy.accuracy)
        self.level_fold, self.level_result, self.level = max(
            ((fold, result, level)
             for fold in folds
             for result in fold.results
             for level in result.levels),
            key=lambda pair: pair[2].tp
        )
        self.top_fold, self.top_result, self.top = max(
            ((fold, result, top)
             for fold in folds
             for result in fold.results
             for top in result.top
             if top.k == 3),
            key=lambda pair: pair[2].top_k_accuracy_score
        )
        self.print()

    def print(self):
        logging.info("fold: %d best f1: %f rule: %s" % (self.f1.fold, self.f1.best_result.f1.f1, self.f1.best_result.f1.rule))
        logging.info("fold: %d best accuracy: %f rule: %s" % (self.accuracy.fold, self.accuracy.best_result.accuracy.accuracy, self.accuracy.best_result.accuracy.rule))
        logging.info("fold: %d name: %s tp: %d rule: %s" % (self.level_fold.fold, self.level.name, self.level.tp, self.level_result.rule))
        logging.info("fold: %d k: %d top: %d rule: %s" % (self.top_fold.fold, self.top.k, self.top.top_k_accuracy_score, self.top_result.rule))