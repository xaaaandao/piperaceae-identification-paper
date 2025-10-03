import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, multilabel_confusion_matrix, confusion_matrix, \
    classification_report, top_k_accuracy_score

from save import save_csv, save_csv_transpose


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

    def save(self, fold, levels, output, rule):
        output_dir = os.path.join(output, rule, "confusion-matrix")
        os.makedirs(output_dir, exist_ok=True)
        self.save_multilabel(fold, levels, output_dir, rule)
        levels = ["%s+%s" % (l.name, l.label) for l in sorted(levels, key=lambda x: x.label)]
        self.save_normalized(fold, levels, output_dir, rule)
        self.save_non_normalized(fold, levels, output_dir, rule)

    def save_multilabel(self, fold, levels, output, rule):
        output_dir = os.path.join(output, "multilabel")
        os.makedirs(output_dir, exist_ok=True)

        for cm, level in zip(self.multilabel, sorted(levels, key=lambda x: x.label)):
            l = "%s+%s" % (level.name, level.label)
            filename = os.path.join(output_dir, "fold-%d-confusion_matrix_multilabel-%s-%s.csv" % (fold, l, rule))
            labels = ["True", "Negative"]
            df = pd.DataFrame(cm, index=labels, columns=labels)
            save_csv(df, filename, header=True, index=True)

    def save_normalized(self, fold, levels, output, rule):
        filename = os.path.join(output, "fold-%d-confusion_matrix_normalized-%s.csv" % (fold, rule))
        df = pd.DataFrame(self.normalized, index=levels, columns=levels)
        save_csv(df, filename, header=True, index=True)

    def save_non_normalized(self, fold, levels, output, rule):
        filename = os.path.join(output, "fold-%d-confusion_matrix_non_normalized-%s.csv" % (fold, rule))
        df = pd.DataFrame(self.non_normalized, index=levels, columns=levels)
        save_csv(df, filename, header=True, index=True)


class Result:
    def __init__(self, fold, levels, output, predict):
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
        self.save(fold, output)

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
        level = max(self.levels, key=lambda x: x.label)
        print("level.label %d" % level.label)
        self.tops = [TopK(k, levels=self.levels, y_score=self.predict.y_score, y_true=self.predict.y_true) for k in
                range(3, level.label)]


    def get_classification_report(self):
        """
        Gera o classification report do experimento.
        :param levels: lista com os levels (classes) utilizadas no experimento.
        :param y_pred: np.ndarray com as classes preditas.
        :param y_true: np.ndarray com as classes verdadeiras.
        :return: dict, com algumas métricas do experimento.
        """
        targets = ["%s+%s" % (l.name, l.label) for l in sorted(self.levels, key=lambda x: x.label)]
        return classification_report(y_pred=self.predict.y_pred, y_true=self.predict.y_true,
                                     labels=np.arange(1, len(self.levels) + 1), target_names=targets, zero_division=0,
                                     output_dict=True)

    def get_true_positive(self):
        tps = np.diag(self.confusion_matrix.non_normalized)
        fns = np.sum(self.confusion_matrix.non_normalized, axis=1) - tps
        fps = np.sum(self.confusion_matrix.non_normalized, axis=0) - tps
        tns = np.sum(self.confusion_matrix.non_normalized) - (np.sum(self.confusion_matrix.non_normalized, axis=1) + np.sum(self.confusion_matrix.non_normalized, axis=0) - tps)
        for tp, fn, fp, tn, level in zip(tps, fns, fps, tns, sorted(self.levels, key=lambda x: x.label)):
            l = list(filter(lambda o: o.label == level.label, self.levels))
            if len(l) > 0:
                l[0].update(tp, fn, fp, tn)


    def save(self, fold, output):
        output_dir = os.path.join(output, "results")
        os.makedirs(output_dir, exist_ok=True)
        self.predict.save(fold, output_dir)
        self.confusion_matrix.save(fold, self.levels, output_dir, self.rule)

        self.save_classification_report(fold, output_dir)
        self.save_f1_accuracy(fold, output_dir)
        self.save_topk(fold, output_dir)
        self.save_tp(fold, output_dir)

    def save_classification_report(self, fold, output):
        output_dir = os.path.join(output, self.rule, "classification_report")
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.join(output_dir, "fold-%d-classification_report-%s.csv" % (fold, self.rule))
        df = pd.DataFrame(self.classification_report)
        df = df.transpose()
        save_csv_transpose(df, filename, header=True, index=False)

    def save_f1_accuracy(self, fold, output):
        output_dir = os.path.join(output, self.rule)
        os.makedirs(output_dir, exist_ok=True)

        data = {
            "metric": ["f1", "accuracy"],
            "value": [self.f1, self.accuracy]
        }
        filename = os.path.join(output_dir, "fold-%d-%s.csv" % (fold, self.rule))
        df = pd.DataFrame(data, columns=data.keys())
        save_csv(df, filename, header=True, index=False)

    def save_topk(self, fold, output):
        output_dir = os.path.join(output, self.rule, "topk")
        os.makedirs(output_dir, exist_ok=True)

        data = {
            "k": [t.k for t in sorted(self.tops, key=lambda x: x.k)],
            "topk_accuracy_score": [t.top_k_accuracy_score for t in sorted(self.tops, key=lambda x: x.k)],
            # "total_test_no_patch": np.repeat(self.fold.total_test_no_patch, len(self.result.topk)),
            # "topk_accuracy_score+100": [topk.top_k_accuracy_score / self.fold.total_test_no_patch for topk in
            #                             sorted(self.result.topk, key=lambda x: x.k)],
            "rule": [self.rule] * len(self.tops) # equivalent a np.repeat, but works in List[str]
        }
        filename = os.path.join(output_dir, "fold-%d-topk-%s.csv" % (fold, self.rule))
        df = pd.DataFrame(data, columns=data.keys())
        save_csv(df, filename, header=True, index=False)

    def save_tp(self, fold, output):
        output_dir = os.path.join(output, self.rule, "true_positive")
        os.makedirs(output_dir, exist_ok=True)

        data = {
            "label": [l.label for l in sorted(self.levels, key=lambda x:x.label)],
            "specific_epithet": [l.name for l in sorted(self.levels, key=lambda x:x.label)],
            "true_positive": [l.tp for l in sorted(self.levels, key=lambda x:x.label)],
            "true_negative": [l.tn for l in sorted(self.levels, key=lambda x:x.label)],
            "false_positive": [l.fp for l in sorted(self.levels, key=lambda x:x.label)],
            "false_negative": [l.fn for l in sorted(self.levels, key=lambda x:x.label)],
        }
        filename = os.path.join(output_dir, "fold-%d-true_positive-%s.csv" % (fold, self.rule))
        df = pd.DataFrame(data, columns=data.keys())
        save_csv(df, filename, header=True, index=False)

