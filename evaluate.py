import logging
import os
import pathlib
from typing import LiteralString

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, multilabel_confusion_matrix, \
    classification_report, top_k_accuracy_score


class TopK:
    def __init__(self, k: int, levels: list = None, topk: float = None, y_score: np.ndarray = None,
                 y_true: np.ndarray = None):
        self.k = k
        # zero are considered None
        self.top_k_accuracy_score = topk if topk is not None else top_k_accuracy_score(y_true=y_true, y_score=y_score,
                                                                                       normalize=False, k=k,
                                                                                       labels=np.arange(1,
                                                                                                        len(levels) + 1))


class Evaluate:
    def __init__(self, levels: list, y_pred: np.ndarray, y_score: np.ndarray, y_true: np.ndarray):
        self.f1 = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
        self.accuracy = accuracy_score(y_pred=y_pred, y_true=y_true)
        self.classification_report = self.set_classification_report(levels, y_pred, y_true)
        self.confusion_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
        self.confusion_matrix_normalized = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true')
        self.confusion_matrix_multilabel = multilabel_confusion_matrix(y_pred=y_pred, y_true=y_true)
        self.topk = self.set_topk(levels, y_score, y_true)

    def get_count_train_label(self, count_train: dict, label: int):
        for count in list(count_train.items()):
            if label == count[0]:
                return count[1]

    def get_count_test_label(self, count_test: dict, label: int):
        for count in list(count_test.items()):
            if label == count[0]:
                return count[1]

    def save_confusion_matrix_multilabel(self, levels: list, output: pathlib.Path | LiteralString | str, rule: str):
        columns = index = ['Positive', 'Negative']
        output = os.path.join(output, 'confusion_matrix', 'multilabel', rule)
        os.makedirs(output, exist_ok=True)
        for idx, confusion_matrix in enumerate(self.confusion_matrix_multilabel, start=1):
            level = list(filter(lambda x: x.label.__eq__(idx), levels))
            if len(level) < 1:
                raise ValueError

            filename = os.path.join(output, 'confusion_matrix_multilabel=%s.csv' % level[0].specific_epithet)
            df = pd.DataFrame(confusion_matrix, index=index, columns=columns)
            df.to_csv(filename, index=index, header=True, sep=';', quoting=2)
            logging.info('Saving %s' % filename)

    def save_confusion_matrix(self, columns: list, index: list, output: pathlib.Path | LiteralString | str, rule: str):
        output = os.path.join(output, 'confusion_matrix')
        os.makedirs(output, exist_ok=True)
        filename = os.path.join(output, 'confusion_matrix+%s.csv' % rule)

        df = pd.DataFrame(self.confusion_matrix, index=index, columns=columns)
        df.to_csv(filename, index=index, header=True, sep=';', quoting=2)
        logging.info('Saving %s' % filename)

    def save_confusion_matrix_normalized(self, columns: list, index: list, output, rule: str):
        output = os.path.join(output, 'confusion_matrix', 'normalized')
        os.makedirs(output, exist_ok=True)
        filename = os.path.join(output, 'confusion_matrix+%s.csv' % rule)

        df = pd.DataFrame(self.confusion_matrix_normalized, index=index, columns=columns)
        df.to_csv(filename, index=index, header=True, sep=';', quoting=2)
        logging.info('Saving %s' % filename)

    def get_index(self, count_train: dict, count_test: dict, levels: list, patch: int):
        return [level.specific_epithet + '(%d-%d)'
                % (int(self.get_count_train_label(count_train, level.label) / patch),
                   int(self.get_count_test_label(count_test, level.label) / patch))
                for level in sorted(levels, key=lambda x: x.label)]

    def save_confusion_matrix_normal_normalized(self, count_train: dict, count_test: dict, levels: list, output,
                                                patch: int, rule: str):
        columns = self.get_columns(levels)
        index = self.get_index(count_train, count_test, levels, patch)
        self.save_confusion_matrix(columns, index, output, rule)
        self.save_confusion_matrix_normalized(columns, index, output, rule)

    def get_columns(self, levels):
        return [level.specific_epithet for level in levels]

    def set_classification_report(self, levels, y_pred, y_true):
        targets = ['label+%s' % (i) for i in range(1, len(levels) + 1)]
        if len(levels) > 0:
            targets = ['%s+%s' % (l.specific_epithet, l.label) for l in sorted(levels, key=lambda x: x.label)]
        return classification_report(y_pred=y_pred, y_true=y_true, labels=np.arange(1, len(levels) + 1),
                                     target_names=targets, zero_division=0,
                                     output_dict=True)

    def save_classification_report(self, output, rule):
        output = os.path.join(output, 'classification_report')
        os.makedirs(output, exist_ok=True)
        filename = os.path.join(output, 'classification_report+%s.csv' % rule)

        df = pd.DataFrame(self.classification_report).transpose()
        df.to_csv(filename, index=True, header=True, sep=';', quoting=2)
        logging.info('Saving %s' % filename)

    def save(self, count_train, count_test, levels, output, patch, rule):
        self.save_confusion_matrix_normal_normalized(count_train, count_test, levels, output, patch, rule)
        self.save_confusion_matrix_multilabel(levels, output, rule)
        self.save_classification_report(output, rule)

    def set_topk(self, levels, y_score, y_true):
        return [TopK(k, levels=levels, y_score=y_score, y_true=y_true) for k in range(3, len(levels))]

    def save_topk(self, count_test, levels, output, rule):
        data = {
            'k': [topk.k for topk in sorted(self.topk, key=lambda x: x.k)],
            'topk_accuracy_score': [topk.top_k_accuracy_score for topk in sorted(self.topk, key=lambda x: x.k)],
            'count_test': np.repeat(count_test, len(self.topk)),
            'topk_accuracy_score+100': [topk.top_k_accuracy_score / count_test for topk in
                                        sorted(self.topk, key=lambda x: x.k)],
            'rule': [rule] * len(self.topk)
        }
        return pd.DataFrame(data, columns=data.keys())

    def save_true_positive(self, count_train, count_test, levels, patch, rule):
        data  = {
            'labels': self.get_index(count_train, count_test, levels, patch),
            'true_positive': list(np.diag(self.confusion_matrix)),
            'rule': [rule] * len(levels)
        }
        return pd.DataFrame(data, columns=data.keys())
