import logging

import numpy as np
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, accuracy_score, f1_score, \
    classification_report

from arrays import max_rule, sum_rule, mult_rule, y_true_no_patch
from level import LevelTP
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

    def get_true_positive(self):
        true_positives = np.diag(self.confusion_matrix)
        for tp, level in zip(true_positives, sorted(self.dataset.levels, key=lambda x: x.label)):
            self.levels.append(LevelTP(level.label, level.specific_epithet, tp))

class BestResult:
    def __init__(self, results):
        f1 = max(results, key=lambda x: x.f1)
        accuracy = max(results, key=lambda x: x.accuracy)
        self.f1 = f1.f1
        self.f1_rule = f1.rule
        self.accuracy = accuracy.accuracy
        self.accuracy_rule = accuracy.rule

        logging.info("Best result F1: %s Rule: %s" % (str(self.f1), self.f1_rule))
        logging.info("Best result accuracy: %s Rule: %s" % (str(self.accuracy), self.accuracy_rule))

class BestFold:
    def __init__(self, folds):
        f1 = max(folds, key=lambda x: x.best_result.f1)
        accuracy = max(folds, key=lambda x: x.best_result.accuracy)
        self.f1 = f1.best_result.f1
        self.f1_fold = f1.fold
        self.f1_rule = f1.best_result.f1_rule
        self.accuracy = accuracy.best_result.accuracy
        self.accuracy_fold = accuracy.fold
        self.accuracy_rule = accuracy.best_result.accuracy_rule

        logging.info("Best fold F1: %s " % self.f1)
        logging.info("Best fold accuracy: %s" % self.accuracy)

class BestMean:
    def __init__(self, means):
        f1 = max(means, key=lambda x: x.f1)
        accuracy = max(means, key=lambda x: x.accuracy)
        self.f1 = f1.f1
        self.f1_std = f1.f1_std
        self.f1_rule = f1.rule
        self.accuracy = accuracy.accuracy
        self.accuracy_std = accuracy.accuracy_std
        self.accuracy_rule = accuracy.rule

        logging.info("Best MEAN result F1: %s Rule: %s" % (str(self.f1), str(self.f1_rule)))
        logging.info("Best MEAN result accuracy: %s Rule: %s" % (str(self.accuracy), str(self.accuracy_rule)))

