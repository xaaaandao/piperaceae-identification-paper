import itertools
import logging

import numpy as np
import pandas as pd


class Mean:
    mean_f1: float
    std_f1: float
    mean_accuracy: float
    std_accuracy: float
    rule: str

    def __init__(self, folds: list, rule: str):
        results = [fold.result for fold in folds]
        predicts = [result.predicts for result in results]
        evaluations = [p.eval for p in list(itertools.chain(*predicts)) if p.rule.__eq__(rule)]
        self.rule = rule
        self.f1(evaluations)
        self.accuracy(evaluations)

    def f1(self, evaluations: list):
        self.mean_f1 = np.mean([evaluation.f1 for evaluation in evaluations])
        self.std_f1 = np.std([evaluation.f1 for evaluation in evaluations])
        logging.info('Mean F1 score %.2f and std F1 score %.2f', self.mean_f1, self.std_f1)

    def accuracy(self, evaluations: list):
        self.mean_accuracy = np.mean([evaluation.accuracy for evaluation in evaluations])
        self.std_accuracy = np.std([evaluation.accuracy for evaluation in evaluations])
        logging.info('Mean accuracy score %.2f and std accuracy score %.2f', self.mean_accuracy, self.std_accuracy)

    def save(self):
        data = {
            'mean': [self.mean_f1, self.mean_accuracy],
            'std': [self.std_f1, self.std_accuracy],
            'metric': ['f1', 'accuracy'],
            'rule': [self.rule, self.rule]
        }
        return pd.DataFrame(data, columns=data.keys())
