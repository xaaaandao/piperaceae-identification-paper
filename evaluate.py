import logging
import os
import pathlib
from typing import LiteralString

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, multilabel_confusion_matrix


class Evaluate:
    def __init__(self, y_pred: np.ndarray, y_score: np.ndarray, y_true: np.ndarray):
        self.f1 = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
        self.accuracy = accuracy_score(y_pred=y_pred, y_true=y_true)
        self.confusion_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
        self.confusion_matrix_normalized = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true')
        self.confusion_matrix_multilabel = multilabel_confusion_matrix(y_pred=y_pred, y_true=y_true)

    def get_count_train_label(self, count_train: dict, label: int):
        print(count_train)
        print(list(count_train.items()))
        for count in list(count_train.items()):
            if label == count[0]:
                return count[1]

    def get_count_test_label(self, count_test: dict, label: int):
        for count in list(count_test.items()):
            if label == count[0]:
                return count[1]

    def save_confusion_matrix_multilabel(self, levels: list, output: pathlib.Path | LiteralString | str):
        columns = index = ['Positive', 'Negative']
        for idx, confusion_matrix in enumerate(self.confusion_matrix_multilabel):
            df = pd.DataFrame(confusion_matrix, index=index, columns=columns)
            # TODO save csv

    def save_confusion_matrix(self, columns: list, index: list, output: pathlib.Path | LiteralString | str, rule: str):
        output = os.path.join(output, 'confusion_matrix')
        os.makedirs(output, exist_ok=True)
        filename = os.path.join(output, 'confusion_matrix+%s.csv' + rule)
        df = pd.DataFrame(self.confusion_matrix, index=index, columns=columns)
        df.to_csv(filename, index=index, header=True, sep=';', quoting=2)
        logging.info('Saving %s' % filename)

    def save_confusion_matrix_normalized(self, columns: list, index: list, output, rule: str):
        output = os.path.join(output, 'confusion_matrix', 'normalized')
        os.makedirs(output, exist_ok=True)
        filename = os.path.join(output, 'confusion_matrix+%s.csv' + rule)
        df = pd.DataFrame(self.confusion_matrix_normalized, index=index, columns=columns)
        df.to_csv(filename, index=index, header=True, sep=';', quoting=2)
        logging.info('Saving %s' % filename)

    def save_confusion_matrix_normal_normalized(self, count_train: dict, count_test: dict, levels: list, output,
                                                rule: str):
        columns = [level.specific_epithet for level in levels]
        index = [level.specific_epithet + \
                 str(self.get_count_train_label(count_train, level.label)) + \
                 str(self.get_count_test_label(count_test, level.label))
                 for level in levels]
        self.save_confusion_matrix(columns, index, output, rule)
        self.save_confusion_matrix_normalized(columns, index, output, rule)

    def save_confusions_matrixs(self, count_train: dict, count_test: dict, levels: list, output, rule: str):
        self.save_confusion_matrix_normal_normalized(count_train, count_test, levels, output, rule)
        self.save_confusion_matrix_multilabel(levels, output)
