import dataclasses
import itertools
import logging
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, multilabel_confusion_matrix

class Result:
    def __init__(self, count_train: dict, count_test: dict, patch: int, predicts: list, time: float):
        self.predicts = predicts
        self.time = time
        self.total_test = np.sum([v for v in count_test.values()])
        self.total_train = np.sum([v for v in count_train.values()])
        self.total_test_no_patch = np.sum([int(v/patch) for v in count_test.values()])
        self.total_train_no_patch = np.sum([int(v/patch) for v in count_train.values()])


class Evaluate:
    def __init__(self, y_pred:np.ndarray, y_score:np.ndarray, y_true:np.ndarray):
        self.f1 = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
        self.accuracy = accuracy_score(y_pred=y_pred, y_true=y_true)
        self.confusion_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
        self.confusion_matrix_normalized = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true')
        self.confusion_matrix_multilabel = multilabel_confusion_matrix(y_pred=y_pred, y_true=y_true)


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
        self.mean_std_f1(evaluations)
        self.mean_std_accuracy(evaluations)

    def mean_std_f1(self, evaluations:list):
        self.mean_f1 = np.mean([evaluation.f1 for evaluation in evaluations])
        self.std_f1 = np.std([evaluation.f1 for evaluation in evaluations])
        logging.info('Mean F1 score %.2f and std F1 score %.2f', self.mean_f1, self.std_f1)

    def mean_std_accuracy(self, evaluations:list):
        self.mean_accuracy = np.mean([evaluation.accuracy for evaluation in evaluations])
        self.std_accuracy = np.std([evaluation.accuracy for evaluation in evaluations])
        logging.info('Mean accuracy score %.2f and std accuracy score %.2f', self.mean_accuracy, self.std_accuracy)