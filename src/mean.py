import collections
import dataclasses
import logging
import os

import numpy as np
import pandas as pd
from numpy import floating

# from save import save_csv


@dataclasses.dataclass
class MeanTop:
    k: int = dataclasses.field(default_factory=int)
    mean: float = dataclasses.field(default_factory=float)
    std: float = dataclasses.field(default_factory=float)


@dataclasses.dataclass
class MeanLevelTP:
    label: int = dataclasses.field(default_factory=int)
    name: str = dataclasses.field(default_factory=str)
    mean_tp: float = dataclasses.field(default_factory=float)
    std_tp: float = dataclasses.field(default_factory=float)


@dataclasses.dataclass
class MeanF1:
    mean: floating = dataclasses.field(default_factory=floating)
    std: floating = dataclasses.field(default_factory=floating)


@dataclasses.dataclass
class MeanAccuracy:
    mean: floating = dataclasses.field(default_factory=floating)
    std: floating = dataclasses.field(default_factory=floating)


class Mean:
    def __init__(self, folds, levels, patch, rule):
        self.rule = rule
        self.accuracy = None
        self.f1 = None
        self.tops : list = []
        self.levels : list = []
        self.get_accuracy(folds)
        self.get_f1(folds)
        self.get_mean_test_label(folds, levels, patch)
        self.get_mean_test(folds, patch)
        self.get_top(folds)
        self.get_tp(folds, levels)
        # self.save(output, patch)

    def get_f1(self, folds):
        mean = np.mean([r.f1 for fold in folds for r in fold.results if r.rule == self.rule])
        std = np.std([r.f1 for fold in folds for r in fold.results if r.rule == self.rule])
        self.f1 = MeanF1(mean, std)
        logging.info("mean f1: %f std: %f rule: %s" % (self.f1.mean, self.f1.std, self.rule))

    def get_accuracy(self, folds):
        mean = np.mean([r.f1 for fold in folds for r in fold.results if r.rule == self.rule])
        std = np.std([r.f1 for fold in folds for r in fold.results if r.rule == self.rule])
        self.accuracy = MeanAccuracy(mean, std)
        logging.info("mean accuracy: %f std: %f rule: %s" % (self.accuracy.mean, self.accuracy.std, self.rule))

    def get_top(self, folds):
        tops = np.array([
            [[top.top_k_accuracy_score for top in sorted(result.tops, key= lambda x: x.k)]
             for result in fold.results if result.rule == self.rule]
            for fold in folds
        ])
        mean_tops = np.mean(tops, axis=(0, 1))
        std_tops = np.std(tops, axis=(0, 1))
        for mean, std, k in zip(mean_tops, std_tops, range(3, len(mean_tops) + 1)):
            self.tops.append(MeanTop(k, mean, std))

    def get_tp(self, folds, levels):
        tps = np.array([
            [[level.tp for level in sorted(result.levels, key= lambda x: x.label)]
             for result in fold.results if result.rule == self.rule]
            for fold in folds
        ])
        mean_level = np.mean(tps, axis=(0, 1))
        std_level = np.std(tps, axis=(0, 1))
        for mean, std, level in zip(mean_level, std_level, sorted(levels, key= lambda x: x.label)):
            self.levels.append(MeanLevelTP(level.label, level.name, mean, std))

    # def save(self, output, patch):
    #     output_dir = os.path.join(output, "means", self.rule)
    #     os.makedirs(output_dir, exist_ok=True)
    #
    #     self.save_f1_accuracy(output_dir)
    #     self.save_top(output_dir, patch)
    #     self.save_tp(output_dir)
    #
    # def save_tp(self, output):
    #     output_dir = os.path.join(output, "tp")
    #     os.makedirs(output_dir, exist_ok=True)
    #
    #     data = self.to_dict_levels()
    #     filename = os.path.join(output_dir, "means_tp-%s.csv" % self.rule)
    #     df = pd.DataFrame(data)
    #     save_csv(df, filename)
    #
    # def save_f1_accuracy(self, output):
    #     data = {
    #         "mean": [self.f1.mean, self.accuracy.mean],
    #         "metric" : ["f1", "accuracy"],
    #         "std": [self.f1.std, self.accuracy.std],
    #         "rule" : [self.rule, self.rule]
    #     }
    #     filename = os.path.join(output, "means-%s.csv" % self.rule)
    #     df = pd.DataFrame(data)
    #     save_csv(df, filename)

    def to_dict_levels(self):
        return {
            "label": [l.label for l in sorted(self.levels, key=lambda x: x.label)],
            "name": [l.name for l in sorted(self.levels, key=lambda x: x.label)],
            "mean": [l.mean_tp for l in sorted(self.levels, key=lambda x: x.label)],
            "std": [l.std_tp for l in sorted(self.levels, key=lambda x: x.label)],
            "mean_test": [self.mean_test_label[l.label] for l in sorted(self.levels, key=lambda x: x.label)],
        }

    # def save_top(self, output, patch):
    #     output_dir = os.path.join(output, "top")
    #     os.makedirs(output_dir, exist_ok=True)
    #
    #     data = self.to_dict_top()
    #     filename = os.path.join(output_dir, "means_top-%s.csv" % self.rule)
    #     df = pd.DataFrame(data)
    #     save_csv(df, filename)

    def to_dict_top(self):
        return {
            "k": [t.k for t in sorted(self.tops, key=lambda x: x.k)],
            "mean": [t.mean for t in sorted(self.tops, key=lambda x: x.k)],
            "std": [t.std for t in sorted(self.tops, key=lambda x: x.k)],
            "mean_test": [self.mean_test] * len(self.tops),
            "mean+100": [t.mean / self.mean_test for t in sorted(self.tops, key=lambda x: x.k)],
        }

    def get_mean_test(self, folds, patch):
        count_test = [np.sum(list(f.test.count.values())) for f in folds]
        self.mean_test = np.mean(count_test) // patch


    def get_mean_test_label(self, folds, levels, patch):
        count_test = [dict(f.test.count) for f in folds]
        soma = collections.Counter()
        count = collections.Counter()
        for d in count_test:
            soma.update(d)
            count.update(d.keys())

        self.mean_test_label = {k: (soma[k] / patch) / count[k] for k in soma}