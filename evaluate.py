import logging
import os
import pathlib
from typing import LiteralString

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, multilabel_confusion_matrix, \
    classification_report, top_k_accuracy_score


class TopK:
    def __init__(self, k: int, levels: list = None, topk: float = None, y_score: np.ndarray = None, y_true: np.ndarray = None):
        self.k = k
        # zero are considered None
        self.top_k_accuracy_score = topk if topk is not None else top_k_accuracy_score(y_true=y_true, y_score=y_score, normalize=False, k=k, labels=np.arange(1, len(levels) + 1))


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
        """
        Encontra a quantidade de treinos de uma determinada classe.
        :param count_train: coleção com todas as quantidades de treinos.
        :param label: classe que deseja encontrar.
        :return: quantidade de treinos de uma determinada classe.
        """
        for count in list(count_train.items()):
            if label == count[0]:
                return count[1]

    def get_count_test_label(self, count_test: dict, label: int):
        """
        Encontra a quantidade de testes de uma determinada classe.
        :param count_test: coleção com todas as quantidades de testes.
        :param label: classe que deseja encontrar.
        :return: quantidade de treinos de uma determinada classe.
        """
        for count in list(count_test.items()):
            if label == count[0]:
                return count[1]

    def save_confusion_matrix_multilabel(self, levels: list, output: pathlib.Path | LiteralString | str, rule: str):
        """
        Salva a matriz de confusão de uma classe em um arquivo CSV.
        :param levels: lista com os levels (classes) da matriz de confusão.
        :param output: local aonde será salvo a matriz de confusão.
        :param rule: regra que foi utilizada naquela matriz de confusão.
        """
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
        """
        Salva a matriz de confusão em um arquivo CSV.
        :param columns: lista com os levels (classes) da matriz de confusão.
        :param index: lista com os levels (classes, quantidade de treinos e teste) da matriz de confusão.
        :param output: local aonde será salvo a matriz de confusão.
        :param rule: regra que foi utilizada naquela matriz de confusão.
        """
        output = os.path.join(output, 'confusion_matrix')
        os.makedirs(output, exist_ok=True)
        filename = os.path.join(output, 'confusion_matrix+%s.csv' % rule)

        df = pd.DataFrame(self.confusion_matrix, index=index, columns=columns)
        df.to_csv(filename, index=index, header=True, sep=';', quoting=2)
        logging.info('Saving %s' % filename)

    def save_confusion_matrix_normalized(self, columns: list, index: list, output: pathlib.Path | LiteralString | str, rule: str):
        """
        Salva a matriz de confusão (valores entre 0 e 1) em um arquivo CSV.
        :param columns: lista com os levels (classes) da matriz de confusão.
        :param index: lista com os levels (classes, quantidade de treinos e teste) da matriz de confusão.
        :param output: local aonde será salvo a matriz de confusão.
        :param rule: regra que foi utilizada naquela matriz de confusão.
        """
        output = os.path.join(output, 'confusion_matrix', 'normalized')
        os.makedirs(output, exist_ok=True)
        filename = os.path.join(output, 'confusion_matrix+%s.csv' % rule)

        df = pd.DataFrame(self.confusion_matrix_normalized, index=index, columns=columns)
        df.to_csv(filename, index=index, header=True, sep=';', quoting=2)
        logging.info('Saving %s' % filename)

    def get_index(self, count_train: dict, count_test: dict, levels: list, patch: int):
        """
        Cria uma lista com os levels (classes), ou seja, nome e a quantidade de treinos e testes, que serão utilizados na matriz de confusão.
        :param count_train: coleção com a quantidade de treinos da matriz de confusão.
        :param count_test: coleção com a quantidade de testes da matriz de confusão.
        :param levels: levels (classes) com nome das espécies utilizadas.
        :param patch: quantidade de divisões da imagem.
        :return: lista com o nome das classes e a quantidade de treinos de testes.
        """
        return [level.specific_epithet + '(%d-%d)'
                % (int(self.get_count_train_label(count_train, level.label) / patch),
                   int(self.get_count_test_label(count_test, level.label) / patch))
                for level in sorted(levels, key=lambda x: x.label)]

    def save_confusion_matrix_normal_normalized(self, count_train: dict, count_test: dict, levels: list, output: pathlib.Path | LiteralString | str, patch: int, rule: str):
        """
        Salva todas as matrizes de confusão.
        :param count_train: coleção com a quantidade de treinos da matriz de confusão.
        :param count_test: coleção com a quantidade de testes da matriz de confusão.
        :param levels: levels (classes) com nome das espécies utilizadas.
        :param patch: quantidade de divisões da imagem.
        :param rule: regra que gerou as matrizes de confusão.
        :return: lista com o nome das classes e a quantidade de treinos de testes.
        """
        columns = self.get_columns(levels)
        index = self.get_index(count_train, count_test, levels, patch)
        self.save_confusion_matrix(columns, index, output, rule)
        self.save_confusion_matrix_normalized(columns, index, output, rule)

    def get_columns(self, levels):
        """
        Cria uma lista com os levels (classes) que serão utilizados na matriz de confusão.
        :param levels: levels (classes) com nome das espécies utilizadas.
        :return: lista com o nome das classes.
        """
        return [level.specific_epithet for level in levels]

    def set_classification_report(self, levels: list, y_pred:np.ndarray, y_true:np.ndarray):
        """
        Gera o classification report do experimento.
        :param levels: lista com os levels (classes) utilizadas no experimento.
        :param y_pred: np.ndarray com as classes preditas.
        :param y_true: np.ndarray com as classes verdadeiras.
        :return: dict, com algumas métricas do experimento.
        """
        targets = ['label+%s' % (i) for i in range(1, len(levels) + 1)]
        if len(levels) > 0:
            targets = ['%s+%s' % (l.specific_epithet, l.label) for l in sorted(levels, key=lambda x: x.label)]
        return classification_report(y_pred=y_pred, y_true=y_true, labels=np.arange(1, len(levels) + 1), target_names=targets, zero_division=0, output_dict=True)

    def save_classification_report(self, output:pathlib.Path | LiteralString | str, rule:str):
        """
        Salva em um arquivo CSV os dados gerados pelo classification report.
        :param levels: lista com os levels (classes) utilizadas no experimento.
        :param rule: regra que gerou o classification report.
        """
        output = os.path.join(output, 'classification_report')
        os.makedirs(output, exist_ok=True)
        filename = os.path.join(output, 'classification_report+%s.csv' % rule)

        df = pd.DataFrame(self.classification_report).transpose()
        df.to_csv(filename, index=True, header=True, sep=';', quoting=2)
        logging.info('Saving %s' % filename)

    def save(self, count_train:dict, count_test:dict, levels:list, output:pathlib.Path|LiteralString|str, patch:int, rule:str):
        """
        Chama as funções que salvam as matrizes de confusão e o classification report.
        :param count_train: coleção com todas as quantidades de treinos.
        :param count_test: coleção com todas as quantidades de testes.
        :param output: local onde será salvo os arquivos.
        :param levels: lista com os levels (classes) utilizadas no experimento.
        :param rule: regra que gerou o classification report.
        """
        self.save_confusion_matrix_normal_normalized(count_train, count_test, levels, output, patch, rule)
        self.save_confusion_matrix_multilabel(levels, output, rule)
        self.save_classification_report(output, rule)

    def set_topk(self, levels:list, y_score:np.ndarray, y_true:np.ndarray):
        """
        Gera uma lista com todas os valores de top k possíveis.
        :param levels: lista com os levels (classes) utilizadas no experimento.
        :param y_pred: np.ndarray com as classes preditas.
        :param y_true: np.ndarray com as classes verdadeiras.
        :return: list, lista com todas os valores de top k.
        """
        return [TopK(k, levels=levels, y_score=y_score, y_true=y_true) for k in range(3, len(levels))]

    def save_topk(self, count_test, levels, output, rule):
        data = {
            'k': [topk.k for topk in sorted(self.topk, key=lambda x: x.k)],
            'topk_accuracy_score': [topk.top_k_accuracy_score for topk in sorted(self.topk, key=lambda x: x.k)],
            'count_test': np.repeat(count_test, len(self.topk)),
            'topk_accuracy_score+100': [topk.top_k_accuracy_score / count_test for topk in
                                        sorted(self.topk, key=lambda x: x.k)],
            'rule': [rule] * len(self.topk) # equivalent a np.repeat, but works in List[str]
        }
        return pd.DataFrame(data, columns=data.keys())

    def save_true_positive(self, count_train:dict, count_test:dict, levels:list, patch:int, rule:str):
        """
        Salva em um arquivo CSV, a quantidade de Verdadeiro Positivos.
        Essa informação é obtida por meio da diagonal principal da matriz de confusão.
        :param count_train: coleção com todas as quantidades de treinos.
        :param count_test: coleção com todas as quantidades de testes.
        :param levels: lista com os levels (classes) utilizadas no experimento.
        :param patch: quantidade de divisões da imagem.
        :param rule: regra que gerou o classification report.
        :return: pd.Dataframe, dataframe com as quantidade de Verdadeiro Positivos de cada classe.
        """
        data = {
            'labels': self.get_index(count_train, count_test, levels, patch),
            'true_positive': list(np.diag(self.confusion_matrix)),
            'rule': [rule] * len(levels)
        }
        return pd.DataFrame(data, columns=data.keys())
