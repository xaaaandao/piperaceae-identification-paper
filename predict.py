import dataclasses
import numpy as np
import pandas as pd

from arrays import max_rule, y_true_no_patch, mult_rule, sum_rule
from dataset import Dataset
from evaluate import Evaluate


class Predict:

    def __init__(self,
                 count_test: int,
                 levels: list,
                 patch: int,
                 rule: str,
                 y_pred_proba: np.ndarray,
                 y_test: np.ndarray,
                 y_pred: np.ndarray = None,
                 y_score: np.ndarray = None,
                 y_true: np.ndarray = None,
                 eval: Evaluate = None
                 ):
        self.count_test = count_test
        self.levels = levels
        self.patch = patch
        self.y_pred_proba = y_pred_proba
        self.rule = rule
        self.y_pred = y_pred
        self.y_score = y_score
        self.y_test = y_test
        self.y_true = y_true
        self.eval = eval
        self.predict_by_rule()
        self.evaluate()

    def predict_by_rule(self):
        match self.rule:
            case 'max':
                self.y_pred, self.y_score = max_rule(self.count_test, len(self.levels), self.patch, self.y_pred_proba)
            case 'sum':
                self.y_pred, self.y_score = sum_rule(self.count_test, len(self.levels), self.patch, self.y_pred_proba)
            case 'mult':
                self.y_pred, self.y_score = mult_rule(self.count_test, len(self.levels), self.patch, self.y_pred_proba)
        self.y_true = y_true_no_patch(self.count_test, self.patch, self.y_test)

    def evaluate(self):
        self.eval = Evaluate(self.y_pred, self.y_score, self.y_true)

    def save(self):
        data = {
            'f1': [self.eval.f1],
            'accuracy': [self.eval.accuracy],
            'rule': [self.rule]
        }
        return pd.DataFrame(data, columns=data.keys())