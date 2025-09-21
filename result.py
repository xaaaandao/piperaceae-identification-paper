import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, accuracy_score, f1_score, \
    classification_report

from arrays import max_rule, sum_rule, mult_rule, y_true_no_patch
from level import get_level_by_label, LevelTP
from topk import TopK


class Result:

    def __init__(self, dataset, rule, y_pred_proba, y_test):
        self.accuracy = float
        self.classification_report = dict()
        self.confusion_matrix = list()
        self.confusion_matrix_normalized = list()
        self.confusion_matrix_multilabel = list()
        self.dataset = dataset
        self.f1 = float
        self.f1_average = "weighted"
        self.levels = list()
        self.rule = rule
        self.y_pred = list | np.ndarray
        self.y_pred_proba = y_pred_proba
        self.y_score = list | np.ndarray
        self.y_test = y_test
        self.y_true = list | np.ndarray

    def get_predictions(self, count_test, n_labels):
        """
        Gera as predições baseado no valor que está no atributo rule.
        Por fim, ele calcula gera o y_true que é um np.ndarray com as classes corretas.
        """
        match self.rule:
            case "max":
                self.y_pred, self.y_score = max_rule(count_test, n_labels, self.dataset.patch, self.y_pred_proba)
            case "sum":
                self.y_pred, self.y_score = sum_rule(count_test, n_labels, self.dataset.patch, self.y_pred_proba)
            case "mult":
                self.y_pred, self.y_score = mult_rule(count_test, n_labels, self.dataset.patch, self.y_pred_proba)
        self.y_true = y_true_no_patch(count_test, self.dataset.patch, self.y_test)
#
    def evaluate(self, n_test, n_labels):
        # TODO necessita do samples.csv
        self.get_predictions(n_test, n_labels)
        self.f1 = f1_score(y_true=self.y_true, y_pred=self.y_pred, average=self.f1_average)
        self.accuracy = accuracy_score(y_pred=self.y_pred, y_true=self.y_true)
        self.classification_report = self.get_classification_report()
        self.confusion_matrix = confusion_matrix(y_true=self.y_true, y_pred=self.y_pred)
        self.confusion_matrix_normalized = confusion_matrix(y_true=self.y_true, y_pred=self.y_pred, normalize="true")
        self.confusion_matrix_multilabel = multilabel_confusion_matrix(y_pred=self.y_pred, y_true=self.y_true)
        self.topk = self.get_topk()
        self.get_true_positive()
#
    def get_topk(self):
        """
        Gera uma lista com todas os valores de top k possíveis.
        :param levels: lista com os levels (classes) utilizadas no experimento.
        :param y_pred: np.ndarray com as classes preditas.
        :param y_true: np.ndarray com as classes verdadeiras.
        :return: list, lista com todas os valores de top k.
        """
        return [TopK(k, levels=self.dataset.levels, y_score=self.y_score, y_true=self.y_true) for k in range(3, self.dataset.n_labels)]

    def get_classification_report(self):
        """
        Gera o classification report do experimento.
        :param levels: lista com os levels (classes) utilizadas no experimento.
        :param y_pred: np.ndarray com as classes preditas.
        :param y_true: np.ndarray com as classes verdadeiras.
        :return: dict, com algumas métricas do experimento.
        """
        targets = ["%s+%s" % (l.specific_epithet, l.label) for l in sorted(self.dataset.levels, key=lambda x: x.label)]
        return classification_report(y_pred=self.y_pred, y_true=self.y_true, labels=np.arange(1, len(self.dataset.levels) + 1), target_names=targets, zero_division=0, output_dict=True)

    def to_dict(self):
        return {
            "f1": self.f1,
            "accuracy": self.accuracy,
            "rule": self.rule
        }

    def save_predictions(self, fold, output):
        output_dir = os.path.join(output, "predictions")
        os.makedirs(output_dir, exist_ok=True)

        self.save_y_pred(fold, output_dir)
        self.save_y_pred_proba(fold, output_dir)
        self.save_y_score(fold, output_dir)

    def save_y_pred(self, fold, output):
        filename = os.path.join(output, "fold-%d-y_pred-%s.npy" % (fold, self.rule))
        np.save(filename, self.y_pred)
        logging.info("saving %s" % filename)

    def save_y_pred_proba(self, fold, output):
        filename = os.path.join(output, "fold-%d-y_pred_proba-%s.npy" % (fold, self.rule))
        np.save(filename, self.y_pred_proba)
        logging.info("saving %s" % filename)

    def save_y_score(self, fold, output):
        filename = os.path.join(output, "fold-%d-y_score-%s.npy" % (fold, self.rule))
        np.save(filename, self.y_score)
        logging.info("saving %s" % filename)

    def save_confusion_matrix(self, fold, output):
        output_dir = os.path.join(output, "confusion_matrix")
        os.makedirs(output_dir, exist_ok=True)

        levels = ["%s+%s" % (l.specific_epithet, l.label) for l in sorted(self.dataset.levels, key=lambda x: x.label)]
        self.save_confusion_matrix_normalized(fold, levels, output_dir)
        self.save_confusion_matrix_non_normalized(fold, levels, output_dir)
        self.save_confusion_matrix_multilabel(fold, output_dir)

    def save_classification_report(self, fold, output):
        output_dir = os.path.join(output, "classification_report")
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.join(output_dir, "fold-%d-classification_report-%s.csv" % (fold, self.rule))
        df = pd.DataFrame(self.classification_report)
        df = df.transpose()
        df.to_csv(filename, sep=";", quoting=2, index=True, header=True, encoding="utf-8")
        logging.info("saving %s" % filename)

    def save_confusion_matrix_normalized(self, fold, levels, output):
        filename = os.path.join(output, "fold-%d-confusion_matrix_normalized-%s.csv" % (fold, self.rule))

        df = pd.DataFrame(self.confusion_matrix_normalized, index=levels, columns=levels)
        df.to_csv(filename, sep=";", quoting=2, index=True, header=True, encoding="utf-8")
        logging.info("saving %s" % filename)

    def save_confusion_matrix_non_normalized(self, fold, levels, output):
        filename = os.path.join(output, "fold-%d-confusion_matrix_non_normalized-%s.csv" % (fold, self.rule))
        df = pd.DataFrame(self.confusion_matrix, index=levels, columns=levels)
        df.to_csv(filename, sep=";", quoting=2, index=True, header=True, encoding="utf-8")
        logging.info("saving %s" % filename)

    def save_confusion_matrix_multilabel(self, fold, output):
        output_dir = os.path.join(output, "multilabel")
        os.makedirs(output_dir, exist_ok=True)

        results = []
        for cm in zip(self.confusion_matrix_multilabel, sorted(self.dataset.levels, key=lambda x: x.label)):
            level = "%s+%s" % (cm[1].specific_epithet, cm[1].label)
            filename = os.path.join(output_dir, "fold-%d-confusion_matrix_multilabel-%s-%s.csv" % (fold, level, self.rule))
            labels = ["True", "Negative"]
            df = pd.DataFrame(cm[0], index=labels, columns=labels)
            df.to_csv(filename, sep=";", quoting=2, index=True, header=True, encoding="utf-8")
            logging.info("saving %s" % filename)

            tp, fp, tn, fn = cm[0].ravel()

            results.append({
                "level": level,
                "true_positive": tp,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "rule": self.rule,
            })

        df = pd.DataFrame(results)
        filename = os.path.join(output, "fold-%d-confusion_matrix_multilabel-%s.csv" % (fold, self.rule))
        df.to_csv(filename, sep=";", quoting=2, index=False, header=True, encoding="utf-8")
        logging.info("saving %s" % filename)

    def save_topk(self, fold, output, total_test_no_patch):
        output_dir = os.path.join(output, "topk")
        os.makedirs(output_dir, exist_ok=True)

        data = {
            "k": [topk.k for topk in sorted(self.topk, key=lambda x: x.k)],
            "topk_accuracy_score": [topk.top_k_accuracy_score for topk in sorted(self.topk, key=lambda x: x.k)],
            "total_test_no_patch": np.repeat(total_test_no_patch, len(self.topk)),
            "topk_accuracy_score+100": [topk.top_k_accuracy_score / total_test_no_patch for topk in
                                        sorted(self.topk, key=lambda x: x.k)],
            "rule": [self.rule] * len(self.topk) # equivalent a np.repeat, but works in List[str]
        }
        filename = os.path.join(output_dir, "fold-%d-topk-%s.csv" % (fold, self.rule))
        df = pd.DataFrame(data, columns=data.keys())
        df.to_csv(filename, sep=";", quoting=2, index=False, encoding="utf-8")
        logging.info("saving %s" % filename)

    def save_tp(self, count_test, fold, output, patch, total_test_no_patch):
        output_dir = os.path.join(output, "true_positive")
        os.makedirs(output_dir, exist_ok=True)

        data = {
            "label": [l.label for l in self.levels],
            "specific_epithet": [l.specific_epithet for l in self.levels],
            "true_positive": [l.true_positive for l in self.levels],
            "count_test": [v / patch for v in dict(sorted(count_test.items())).values()],
        }
        filename = os.path.join(output_dir, "fold-%d-true_positive-%s.csv" % (fold, self.rule))
        df = pd.DataFrame(data, columns=data.keys())
        df.to_csv(filename, sep=";", quoting=2, index=False, encoding="utf-8")
        logging.info("saving %s" % filename)

    def get_true_positive(self):
        true_positives = np.diag(self.confusion_matrix)
        for tp, level in zip(true_positives, sorted(self.dataset.levels, key=lambda x: x.label)):
            self.levels.append(LevelTP(level.label, level.specific_epithet, tp))