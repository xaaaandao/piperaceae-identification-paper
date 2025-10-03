import logging
import os

import pandas as pd

from save import save_csv


class BestResult:
    def __init__(self, results):
        self.f1 = max(results, key=lambda x: x.f1)
        self.accuracy = max(results, key=lambda x: x.accuracy)
        self.level_result, self.level = max(
            ((result, level) for result in results for level in result.levels),
            key=lambda pair: pair[1].tp
        )
        self.top_result, self.top = max(
            ((result, top) for result in results for top in result.tops if top.k == 3),
            key=lambda pair: pair[1].top_k_accuracy_score
        )
        self.print()

    def print(self):
        logging.info("BEST f1: %f rule: %s" % (self.f1.f1, self.f1.rule))
        logging.info("BEST accuracy: %f rule: %s" % (self.accuracy.accuracy, self.accuracy.rule))
        logging.info("BEST name: %s tp: %d rule: %s" % (self.level.name, self.level.tp, self.level_result.rule))
        logging.info("BEST k: %d top: %d rule: %s" % (self.top.k, self.top.top_k_accuracy_score, self.top_result.rule))

    def save(self, output):
        data = {
            "best": [self.f1.f1, self.accuracy.accuracy],
            "metric" : ["f1", "accuracy"],
            "rule" : [self.f1.rule, self.accuracy.rule]
        }
        filename = os.path.join(output, "best-result.csv")
        df = pd.DataFrame(data)
        save_csv(df, filename)


class BestMean:
    def __init__(self, means):
        self.f1 = max(means, key=lambda m: m.f1.mean)
        self.accuracy = max(means, key=lambda m: m.accuracy.mean)
        self.mean_top, self.top = max(
            ((mean, top) for mean in means for top in mean.tops if top.k == 3),
            key=lambda pair: pair[1].mean
        )
        self.print()

    def print(self):
        logging.info("BEST f1: %f std: %f rule: %s" % (self.f1.f1.mean, self.f1.f1.std, self.f1.rule))
        logging.info("BEST accuracy: %f std: %f rule: %s" % (self.accuracy.accuracy.mean, self.accuracy.accuracy.std, self.accuracy.rule))
        logging.info("BEST k: %d top : %d std: %d rule: %s" % (self.top.k, self.top.mean, self.top.std, self.mean_top.rule))

    def save(self, output):
        data = {
            "mean": [self.f1.f1.mean, self.accuracy.accuracy.mean],
            "metric" : ["f1", "accuracy"],
            "std": [self.f1.f1.std, self.accuracy.accuracy.std],
            "rule" : [self.f1.rule, self.accuracy.rule]
        }
        filename = os.path.join(output, "best-mean.csv")
        df = pd.DataFrame(data)
        save_csv(df, filename)

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
             for top in result.tops
             if top.k == 3),
            key=lambda pair: pair[2].top_k_accuracy_score
        )
        self.print()

    def print(self):
        logging.info("fold: %d best f1: %f rule: %s" % (self.f1.fold, self.f1.best_result.f1.f1, self.f1.best_result.f1.rule))
        logging.info("fold: %d best accuracy: %f rule: %s" % (self.accuracy.fold, self.accuracy.best_result.accuracy.accuracy, self.accuracy.best_result.accuracy.rule))
        logging.info("fold: %d name: %s tp: %d rule: %s" % (self.level_fold.fold, self.level.name, self.level.tp, self.level_result.rule))
        logging.info("fold: %d k: %d top: %d rule: %s" % (self.top_fold.fold, self.top.k, self.top.top_k_accuracy_score, self.top_result.rule))

    def save(self, output):
        data = {
            "fold" : [self.f1.fold, self.accuracy.fold],
            "metric" : ["f1", "accuracy"],
            "value" : [self.f1.best_result.f1.f1, self.accuracy.best_result.accuracy.accuracy],
            "rule" : [self.f1.best_result.f1.rule, self.accuracy.best_result.accuracy.rule],
        }
        filename = os.path.join(output, "best-fold.csv")
        df = pd.DataFrame(data)
        save_csv(df, filename)
