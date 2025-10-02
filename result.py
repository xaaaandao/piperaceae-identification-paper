import logging

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, multilabel_confusion_matrix, confusion_matrix, \
    classification_report, top_k_accuracy_score

class Mean:
    def __init__(self, folds, rule):
        self.rule = rule
        self.accuracy = 0
        self.accuracy_std = 0
        self.f1 = 0
        self.f1_std = 0
        self.top = list
        self.top_std = list
        self.tp = list
        self.tp_std = list
        self.get_accuracy(folds)
        self.get_f1(folds)
        self.get_top(folds)
        self.get_tp(folds)

    def get_f1(self, folds):
        self.f1 = np.mean([r.f1 for fold in folds for r in fold.results if r.rule == self.rule])
        self.f1_std = np.std([r.f1 for fold in folds for r in fold.results if r.rule == self.rule])
        logging.info("mean f1: %f std: %f rule: %s" % (self.f1, self.f1_std, self.rule))

    def get_accuracy(self, folds):
        self.accuracy = np.mean([r.accuracy for fold in folds for r in fold.results if r.rule == self.rule])
        self.accuracy_std = np.std([r.accuracy for fold in folds for r in fold.results if r.rule == self.rule])
        logging.info("mean accuracy: %f std %f rule: %s" % (self.accuracy, self.accuracy_std, self.rule))

    def get_top(self, folds):
        pass

    def get_tp(self, folds):
        pass


class TopK:
    def __init__(self, k: int, levels: list = None, topk: float = None, y_score: np.ndarray = None, y_true: np.ndarray = None):
        self.k = k
        # zero are considered None
        self.top_k_accuracy_score = topk if topk is not None else top_k_accuracy_score(y_true=y_true, y_score=y_score, normalize=False, k=k, labels=np.arange(1, len(levels) + 1))

class Predict:
    def __init__(self, patch, rule, y_pred_proba, y_test):
        self.num_groups = y_pred_proba.shape[0] // patch
        self.patch = patch
        self.rule = rule
        self.y_pred = []
        self.y_score = []
        self.y_pred_proba = y_pred_proba
        self.y_test = y_test
        self.y_true = []
        self.get_predictions()

    def get_predictions(self):
        """
        Gera as predições baseado no valor que está no atributo rule.
        Por fim, ele calcula gera o y_true que é um np.ndarray com as classes corretas.
        """
        match self.rule:
            case "max":
                self.max_rule()
            case "sum":
                self.sum_rule()
            case "mult":
                self.mult_rule()

        self.y_true_no_patch()

    def sum_rule(self):
        pos = []
        sums = []
        for i in range(self.num_groups):
            start_idx = i * self.patch
            end_idx = start_idx + self.patch
            group_sum = self.y_pred_proba[start_idx:end_idx].sum(axis=0)

            sums.append(group_sum)
            pos.append(np.argmax(group_sum)+1)

        self.y_pred = np.array(pos)
        self.y_score = np.array(sums)

    def mult_rule(self):
        pos = []
        mults = []
        for i in range(self.num_groups):
            start_idx = i * self.patch
            end_idx = start_idx + self.patch
            group = self.y_pred_proba[start_idx:end_idx]
            group_mult = group.prod(axis=0)

            mults.append(group_mult)
            pos.append(np.argmax(group_mult)+1)

        self.y_pred = np.array(pos)
        self.y_score = np.array(mults)

    def max_rule(self):
        pos = []
        maxs = []
        for i in range(self.num_groups):
            start_idx = i * self.patch
            end_idx = start_idx + self.patch
            group = self.y_pred_proba[start_idx:end_idx]
            max_idx = np.argmax(group)
            row_in_group, col = np.unravel_index(max_idx, group.shape)

            pos.append(col+1)
            maxs.append(group[row_in_group])

        self.y_pred = np.array(pos)
        self.y_score = np.array(maxs)

    def y_true_no_patch(self):
        labels = []
        num_groups = self.y_pred_proba.shape[0] // self.patch
        for i in range(num_groups):
            start_idx = i * self.patch
            end_idx = start_idx + self.patch
            group = self.y_test[start_idx:end_idx]
            labels.append(group[0])

        self.y_true = np.array(labels)


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
        level = max(self.levels, key=lambda x: x.label)
        self.top = [TopK(k, levels=self.levels, y_score=self.predict.y_score, y_true=self.predict.y_true) for k in
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


class BestResult:
    def __init__(self, results):
        self.f1 = max(results, key=lambda x: x.f1)
        self.accuracy = max(results, key=lambda x: x.accuracy)
        self.level_result, self.level = max(
            ((result, level) for result in results for level in result.levels),
            key=lambda pair: pair[1].tp
        )
        self.top_result, self.top = max(
            ((result, top) for result in results for top in result.top if top.k == 3),
            key=lambda pair: pair[1].top_k_accuracy_score
        )
        self.print()

    def print(self):
        logging.info("best f1: %f rule: %s" % (self.f1.f1, self.f1.rule))
        logging.info("best accuracy: %f rule: %s" % (self.accuracy.accuracy, self.accuracy.rule))
        logging.info("best name: %s tp: %d rule: %s" % (self.level.name, self.level.tp, self.level_result.rule))
        logging.info("best k: %d top: %d rule: %s" % (self.top.k, self.top.top_k_accuracy_score, self.top_result.rule))