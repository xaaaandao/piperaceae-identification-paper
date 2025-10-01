import numpy as np
from sklearn.metrics import f1_score, accuracy_score, multilabel_confusion_matrix, confusion_matrix, \
    classification_report

from arrays import y_true_no_patch, max_rule, sum_rule, mult_rule


class Predict:
    def __init__(self, patch, rule, y_pred_proba, y_test):
        self.patch = patch
        self.rule = rule
        self.y_pred = []
        self.y_score = []
        self.y_pred_proba = y_pred_proba
        self.y_test = y_test
        self.y_true = []
        qtd_test, qtd_labels = y_pred_proba.shape
        self.get_predictions(qtd_labels, qtd_test)

    def get_predictions(self, qtd_labels, qtd_test):
        """
        Gera as predições baseado no valor que está no atributo rule.
        Por fim, ele calcula gera o y_true que é um np.ndarray com as classes corretas.
        """
        match self.rule:
            case "max":
                self.y_pred, self.y_score = max_rule(qtd_test, qtd_labels, self.patch, self.y_pred_proba)
            case "sum":
                self.y_pred, self.y_score = sum_rule(qtd_test, qtd_labels, self.patch, self.y_pred_proba)
            case "mult":
                self.y_pred, self.y_score = mult_rule(qtd_test, qtd_labels, self.patch, self.y_pred_proba)
        self.y_true = y_true_no_patch(qtd_test, self.patch, self.y_test)

class ConfusionMatrix:
    def __init__(self, y_true, y_pred):
        self.multilabel = multilabel_confusion_matrix(y_pred=y_pred, y_true=y_true)
        self.non_normalized = confusion_matrix(y_true=y_true, y_pred=y_pred)
        self.normalized = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize="true")

class Result:

    def __init__(self, levels, predict):
        self.accuracy = float
        self.classification_report = dict()
        self.confusion_matrix = ConfusionMatrix(predict.y_score, predict.y_pred)
        self.f1 = float
        self.f1_average = "weighted"
        self.levels = levels
        self.predict = predict
        self.rule = self.predict.rule
        self.evaluate()

    def evaluate(self):
        self.f1 = f1_score(y_true=self.predict.y_true, y_pred=self.predict.y_pred, average=self.f1_average)
        self.accuracy = accuracy_score(y_pred=self.predict.y_pred, y_true=self.predict.y_true)
        self.classification_report = self.get_classification_report()

    # def get_topk(self):
    #     """
    #     Gera uma lista com todas os valores de top k possíveis.
    #     :param levels: lista com os levels (classes) utilizadas no experimento.
    #     :param y_pred: np.ndarray com as classes preditas.
    #     :param y_true: np.ndarray com as classes verdadeiras.
    #     :return: list, lista com todas os valores de top k.
    #     """
    #     return [TopK(k, levels=self.dataset.levels, y_score=self.y_score, y_true=self.y_true) for k in range(3, self.dataset.n_labels)]
    #
    def get_classification_report(self):
        """
        Gera o classification report do experimento.
        :param levels: lista com os levels (classes) utilizadas no experimento.
        :param y_pred: np.ndarray com as classes preditas.
        :param y_true: np.ndarray com as classes verdadeiras.
        :return: dict, com algumas métricas do experimento.
        """
        targets = ["%s+%s" % (l.specific_epithet, l.label) for l in sorted(self.levels, key=lambda x: x.label)]
        return classification_report(y_pred=self.predict.y_pred, y_true=self.predict.y_true,
                                     labels=np.arange(1, len(self.levels) + 1), target_names=targets, zero_division=0,
                                     output_dict=True)

    # def get_true_positive(self):
    #     true_positives = np.diag(self.confusion_matrix)
    #     for tp, level in zip(true_positives, sorted(self.dataset.levels, key=lambda x: x.label)):
    #         self.levels.append(LevelTP(level.label, level.specific_epithet, tp))
