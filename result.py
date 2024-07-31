import itertools
import logging
import os
import pathlib
from typing import LiteralString

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, accuracy_score, f1_score, \
    classification_report

from arrays import sum_rule, max_rule, mult_rule, y_true_no_patch
from evaluate import TopK


class Result:
    def __init__(self,
                 count_test: int,
                 levels: list,
                 patch: int,
                 rule: str,
                 y_pred_proba: np.ndarray,
                 y_test: np.ndarray):
        self.count_test = count_test
        self.levels = levels
        self.patch = patch
        self.rule = rule
        self.y_pred_proba = y_pred_proba
        self.y_test = y_test
        self.y_pred = None
        self.y_score = None
        self.y_true = None
        self.f1 = None
        self.accuracy = None
        self.classification_report = None
        self.confusion_matrix = None
        self.confusion_matrix_normalized = None
        self.confusion_matrix_multilabel = None
        self.topk = None
        self.apply_rules()
        self.evaluate()

    def apply_rules(self):
        """
        Gera as predições baseado no valor que está no atributo rule.
        Por fim, ele calcula gera o y_true que é um np.ndarray com as classes corretas.
        """
        match self.rule:
            case 'max':
                self.y_pred, self.y_score = max_rule(self.count_test, len(self.levels), self.patch, self.y_pred_proba)
            case 'sum':
                self.y_pred, self.y_score = sum_rule(self.count_test, len(self.levels), self.patch, self.y_pred_proba)
            case 'mult':
                self.y_pred, self.y_score = mult_rule(self.count_test, len(self.levels), self.patch, self.y_pred_proba)
        self.y_true = y_true_no_patch(self.count_test, self.patch, self.y_test)

    def evaluate(self):
        self.f1 = f1_score(y_true=self.y_true, y_pred=self.y_pred, average='weighted')
        self.accuracy = accuracy_score(y_pred=self.y_pred, y_true=self.y_true)
        self.classification_report = self.set_classification_report(self.levels, self.y_pred, self.y_true)
        self.confusion_matrix = confusion_matrix(y_true=self.y_true, y_pred=self.y_pred)
        self.confusion_matrix_normalized = confusion_matrix(y_true=self.y_true, y_pred=self.y_pred, normalize='true')
        self.confusion_matrix_multilabel = multilabel_confusion_matrix(y_pred=self.y_pred, y_true=self.y_true)
        self.topk = self.set_topk(self.levels, self.y_score, self.y_true)

    def set_topk(self, levels: list, y_score: np.ndarray, y_true: np.ndarray):
        """
        Gera uma lista com todas os valores de top k possíveis.
        :param levels: lista com os levels (classes) utilizadas no experimento.
        :param y_pred: np.ndarray com as classes preditas.
        :param y_true: np.ndarray com as classes verdadeiras.
        :return: list, lista com todas os valores de top k.
        """
        return [TopK(k, levels=levels, y_score=y_score, y_true=y_true) for k in range(3, len(levels))]

    def set_classification_report(self, levels: list, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Gera o classification report do experimento.
        :param levels: lista com os levels (classes) utilizadas no experimento.
        :param y_pred: np.ndarray com as classes preditas.
        :param y_true: np.ndarray com as classes verdadeiras.
        :return: dict, com algumas métricas do experimento.
        """
        targets = ['label+%s' % (i) for i in range(1, len(self.levels) + 1)]
        if len(self.levels) > 0:
            targets = ['%s+%s' % (l.specific_epithet, l.label) for l in sorted(self.levels, key=lambda x: x.label)]
        return classification_report(y_pred=self.y_pred, y_true=self.y_true, labels=np.arange(1, len(self.levels) + 1),
                                     target_names=targets, zero_division=0, output_dict=True)
