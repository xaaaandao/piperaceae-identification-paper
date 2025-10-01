import dataclasses
import itertools

import numpy as np

from level import Level, get_level_by_name, get_level_by_label


@dataclasses.dataclass
class MeanTopK:
    k: int
    mean: float
    std: float

class MeanTP(Level):
    def __init__(self, label, mean, specific_epithet, std):
        super().__init__(label, specific_epithet)
        self.mean = mean
        self.std = std

class Mean:
    def __init__(self, folds, rule):
        self.f1 = 0
        self.f1_std = 0
        self.accuracy = 0
        self.accuracy_std = 0
        self.topks = list()
        self.true_positives = list()
        self.folds = folds
        self.rule = rule

        self.results = [result for fold in self.folds for result in fold.results if result.rule == self.rule]
        self.get_f1()
        self.get_accuracy()
        self.get_topk()
        self.get_true_positives()

    def get_f1(self):
        f1s = [result.f1 for result in self.results]
        self.f1 = np.mean(f1s)
        self.f1_std = np.std(f1s)

    def get_accuracy(self):
        accuracies = [result.accuracy for result in self.results]
        self.accuracy = np.mean(accuracies)
        self.accuracy_std = np.std(accuracies)

    def to_dict(self):
        return {
            "f1": self.f1,
            "f1_std": self.f1_std,
            "accuracy": self.accuracy,
            "accuracy_std": self.accuracy_std,
            "rule": self.rule,
        }

    def get_topk(self):
        t = [ result.topk for result in self.results ]
        t = list(itertools.chain(*t))
        max_k = max(t, key=lambda x: x.k).k
        for k in range(3, max_k + 1):
            values = [topk.top_k_accuracy_score for topk in t if topk.k == k]
            self.topks.append(MeanTopK(k, np.mean(values), np.std(values)))

    def get_true_positives(self):
        tps = [ result.levels for result in self.results ]
        tps = list(itertools.chain(*tps))
        min_label = min(tps, key=lambda x: x.label)
        max_label = max(tps, key=lambda x: x.label)
        for label in range(min_label.label, max_label.label + 1):
            values = [t.true_positive for t in tps if t.label == label]
            level = get_level_by_label(label, tps)
            self.true_positives.append(MeanTP(level.label, np.mean(values), level.specific_epithet, np.std(values)))
