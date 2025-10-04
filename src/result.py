import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, multilabel_confusion_matrix, confusion_matrix, \
    classification_report, top_k_accuracy_score

# from save import save_csv, save_csv_transpose


class TopK:
    def __init__(self, k: int, levels: list = None, y_score: np.ndarray = None, y_true: np.ndarray = None):
        self.k = k
        # zero are considered None
        self.top_k_accuracy_score = top_k_accuracy_score(y_true=y_true, y_score=y_score, normalize=False, k=k, labels=np.arange(1, len(levels) + 1))


class ConfusionMatrix:
    def __init__(self, y_true, y_pred):
        self.multilabel = multilabel_confusion_matrix(y_pred=y_pred, y_true=y_true)
        self.non_normalized = confusion_matrix(y_true=y_true, y_pred=y_pred)
        self.normalized = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize="true")

class Result:
    def __init__(self, levels, predict):
        self.accuracy = float
        self.classification_report = dict()
        self.confusion_matrix = ConfusionMatrix(predict.y_true, predict.y_pred)
        self.f1 = float
        self.f1_average = "weighted"
        self.levels = levels.copy()
        self.predict = predict
        self.rule = self.predict.rule
        self.tops = list
        self.evaluate()

    def evaluate(self):
        self.f1 = f1_score(y_true=self.predict.y_true, y_pred=self.predict.y_pred, average=self.f1_average)
        self.accuracy = accuracy_score(y_pred=self.predict.y_pred, y_true=self.predict.y_true)
        self.classification_report = self.get_classification_report()
        self.get_top()
        self.get_true_positive()

    def get_top(self):
        """
        Gera uma lista com todas os valores de top k possíveis.
        :param levels: lista com os levels (classes) utilizadas no experimento.
        :param y_pred: np.ndarray com as classes preditas.
        :param y_true: np.ndarray com as classes verdadeiras.
        :return: list, lista com todas os valores de top k.
        """
        self.tops = [TopK(k, levels=self.levels, y_score=self.predict.y_score, y_true=self.predict.y_true) for k in
            range(3, len(self.levels))]


    def get_classification_report(self):
        """
        Gera o classification report do experimento.
        :param levels: lista com os levels (classes) utilizadas no experimento.
        :param y_pred: np.ndarray com as classes preditas.
        :param y_true: np.ndarray com as classes verdadeiras.
        :return: dict, com algumas métricas do experimento.
        """
        targets = ["%s+%s" % (l.name, l.label) for l in sorted(self.levels, key=lambda x: x.label)]
        return classification_report(y_pred=self.predict.y_pred, y_true=self.predict.y_true, labels=np.arange(1, len(self.levels) + 1), target_names=targets, zero_division=0, output_dict=True)

    def get_true_positive(self):
        tps = np.diag(self.confusion_matrix.non_normalized)
        fns = np.sum(self.confusion_matrix.non_normalized, axis=1) - tps
        fps = np.sum(self.confusion_matrix.non_normalized, axis=0) - tps
        tns = np.sum(self.confusion_matrix.non_normalized) - (np.sum(self.confusion_matrix.non_normalized, axis=1) + np.sum(self.confusion_matrix.non_normalized, axis=0) - tps)
        for tp, fn, fp, tn, level in zip(tps, fns, fps, tns, sorted(self.levels, key=lambda x: x.label)):
            l = list(filter(lambda o: o.label == level.label, self.levels))
            if len(l) > 0:
                l[0].update(tp, fn, fp, tn)


    #     self.confusion_matrix.save(fold, self.levels, output_dir, self.rule)


    def to_dict_metrics(self):
        return {
            "metric": ["f1", "accuracy"],
            "value": [self.f1, self.accuracy]
        }

    def to_dict_top(self):
        return {
            "k": [t.k for t in sorted(self.tops, key=lambda x: x.k)],
            "topk_accuracy_score": [t.top_k_accuracy_score for t in sorted(self.tops, key=lambda x: x.k)],
            # "total_test_no_patch": [self.test.total_no_patch]  * len(self.tops),
            # "topk_accuracy_score+100": [topk.top_k_accuracy_score // self.test.total_no_patch for topk in sorted(self.tops, key=lambda x: x.k)],
            "rule": [self.rule] * len(self.tops)  # equivalent a np.repeat, but works in List[str]
        }

    def to_dict_tp(self):
        return {
            "label": [l.label for l in sorted(self.levels, key=lambda x: x.label)],
            "specific_epithet": [l.name for l in sorted(self.levels, key=lambda x: x.label)],
            "true_positive": [l.tp for l in sorted(self.levels, key=lambda x: x.label)],
            "true_negative": [l.tn for l in sorted(self.levels, key=lambda x: x.label)],
            "false_positive": [l.fp for l in sorted(self.levels, key=lambda x: x.label)],
            "false_negative": [l.fn for l in sorted(self.levels, key=lambda x: x.label)],
        }