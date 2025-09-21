import collections
import functools
import itertools
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from classifiers import get_classifier
from dataset import Dataset
from fold import Fold
from mean import Mean
from save import save_csv_transpose


class Experiment:
    def __init__(self, classifier: Any, dataset: Dataset, backend: str = "loky",
                 cv_metric: str = "f1_weighted", folds: int = 5,
                 metrics: list[str] = None, n_jobs: int = -1,
                 seed: int = 1234, verbose: int = 42):
        if metrics is None:
            metrics = ["f1", "accuracy"]

        self.backend = backend
        self.best_fold_f1 = None
        self.best_fold_accuracy = None
        self.best_mean_f1 = None
        self.best_mean_accuracy = None
        self.cv_metric = cv_metric
        self.dataset = dataset
        self.folds = folds
        self.indexes = None
        self.means = None
        self.metrics = metrics
        self.n_jobs = n_jobs
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
            fold.run(self.backend, self.classifier, **kwargs)

        self.best_fold_f1 = max(folds, key=lambda x: x.best_f1.f1)
        self.best_fold_accuracy = max(folds, key=lambda x: x.best_accuracy.accuracy)
        logging.info("Best fold F1: %s" % self.best_fold_f1.fold)
        logging.info("Best fold accuracy: %s" % self.best_fold_accuracy.fold)

        self.means = [Mean(folds, rule) for rule in ["sum", "max", "mult"]]
        self.best_mean_f1 = max(self.means, key=lambda x: x.f1)
        self.best_mean_accuracy = max(self.means, key=lambda x: x.accuracy)
        logging.info("Best MEAN result F1: %s Rule: %s" % (str(self.best_mean_f1.f1), self.best_mean_f1.rule))
        logging.info("Best MEAN result accuracy: %s Rule: %s" % (str(self.best_mean_accuracy.accuracy), self.best_mean_f1.rule))

        self.save(folds, output)

    def save(self, folds, output):
        self.save_best(output)
        self.save_experiement(output)
        self.save_folds(folds, output)
        self.save_mean(folds, output)

    def save_folds(self, folds, output):
        for f in folds:
            output_dir = os.path.join(output, "fold-%d" % f.fold)
            os.makedirs(output_dir, exist_ok=True)
            f.save(output_dir)

    def save_mean(self, folds, output):
        output_dir = os.path.join(output, "mean")
        os.makedirs(output_dir, exist_ok=True)

        self.save_mean_f1_accuracy(output_dir)
        self.save_mean_topk(folds, output_dir)
        self.save_mean_true_positive(folds, output_dir)

    def save_mean_f1_accuracy(self, output):
        filename = os.path.join(output, "means.csv")
        df = pd.DataFrame([mean.to_dict() for mean in self.means])
        df.to_csv(filename, index=False)
        logging.info("saving %s" % filename)

    def save_best(self, output):
        output_dir = os.path.join(output, "best")
        os.makedirs(output_dir, exist_ok=True)

        self.save_best_mean(output_dir)
        self.save_best_fold(output_dir)

    def save_best_mean(self, output):
        filename = os.path.join(output, "best_means.csv")
        data = {
            "best_f1": [self.best_mean_f1.f1],
            "best_f1_std": [self.best_mean_f1.f1_std],
            "best_f1_rule": [self.best_mean_f1.rule],
            "best_accuracy": [self.best_mean_accuracy.accuracy],
            "best_accuracy_std": [self.best_mean_accuracy.accuracy_std],
            "best_accuracy_rule": [self.best_mean_accuracy.rule]
        }
        save_csv_transpose(data, filename)

    def save_best_fold(self, output):
        filename = os.path.join(output, "best_fold.csv")
        data = {
            "best_fold_f1": [self.best_fold_f1.best_f1.f1],
            "best_fold_f1_rule": [self.best_fold_f1.best_f1.rule],
            "best_fold_f1_fold": [self.best_fold_f1.fold],
            "best_fold_accuracy": [self.best_fold_accuracy.best_accuracy.accuracy],
            "best_fold_accuracy_rule": [self.best_fold_accuracy.best_accuracy.rule],
            "best_fold_accuracy_fold": [self.best_fold_accuracy.fold],
        }
        save_csv_transpose(data, filename)

    def save_experiement(self, output):
        filename = os.path.join(output, "experiment.csv")
        data = {
            "backend": [self.backend],
            "classifier": [self.classifier.__class__.__name__],
            "cv_metric": [self.cv_metric],
            "data_aug": [self.dataset.data_aug],
            "folds": [self.folds],
            "input": [self.dataset.input_dir],
            "model": [self.dataset.model],
            "metrics": [str(self.metrics)],
            "n_features": [self.dataset.n_features],
            "n_jobs": [self.n_jobs],
            "n_label": [self.dataset.n_labels],
            "n_samples": [self.dataset.n_samples],
            "seed": [self.seed],
            "verbose": [self.verbose]
        }
        save_csv_transpose(data, filename)

    def save_mean_topk(self, folds, output):
        output_dir = os.path.join(output, "topk")
        os.makedirs(output_dir, exist_ok=True)
        mean_test = [f.total_test_no_patch for f in folds]

        for rule in ["sum", "max", "mult"]:
            filename = os.path.join(output_dir , "means+topk+%s.csv" % rule)
            topks = [m.topks for m in self.means if m.rule == rule]
            topks = list(itertools.chain(*topks))
            data = {
                "k": [t.k for t in topks],
                "top_k_accuracy_score": [t.mean for t in topks],
                "top_k_accuracy_score_std": [t.std for t in topks],
                "mean_test": np.mean(mean_test),
            }
            df = pd.DataFrame(data)
            df.to_csv(filename, sep=";", quoting=2, index=False, encoding="utf-8")
            logging.info("saving %s" % filename)

    def save_mean_true_positive(self, folds, output):
        output_dir = os.path.join(output, "true_positive")
        os.makedirs(output_dir, exist_ok=True)
        tests = [f.count_test for f in folds]
        count_test = dict(functools.reduce(lambda x, y: collections.Counter(x) + collections.Counter(y), tests))

        for rule in ["sum", "max", "mult"]:
            filename = os.path.join(output_dir , "means+true_positive+%s.csv" % rule)
            tps = [m.true_positives for m in self.means if m.rule == rule]
            tps = list(itertools.chain(*tps))
            data = {
                "label": [t.label for t in tps],
                "specific_epithet": [t.specific_epithet for t in tps],
                "true_positive": [t.mean for t in tps],
                "true_positive_std": [t.std for t in tps],
                "mean_test": [(count_test[t.label] / self.dataset.patch) / len(tests) for t in tps],
            }
            df = pd.DataFrame(data)
            df.to_csv(filename, sep=";", quoting=2, index=False, encoding="utf-8")
            logging.info("saving %s" % filename)
