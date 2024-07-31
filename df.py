import itertools
import logging
import os
import pathlib
from typing import LiteralString

import numpy as np
import pandas as pd

from dataset import Dataset


def best_evals(df_evaluations):
    df_accuracy = df_evaluations.loc[df_evaluations['accuracy'].idxmax()]
    df_f1 = df_evaluations.loc[df_evaluations['f1'].idxmax()]
    data = {'metric': ['accuracy', 'f1'],
            'value': [df_accuracy['accuracy'], df_f1['f1']],
            'rule': [df_accuracy['rule'], df_f1['rule']]}
    return pd.DataFrame(data, columns=list(data.keys()), index=None)


def classifications(predicts):
    return {p.rule: pd.DataFrame(p.classification_report).transpose() for p in predicts}


def confusion_matrix(count_train, count_test, dataset, predicts):
    columns = get_columns(dataset.levels)
    index = get_index(count_train, count_test, dataset)
    return {p.rule: pd.DataFrame(p.confusion_matrix, index=index, columns=columns) for p in predicts}


def confusion_matrix_normalized(count_train, count_test, dataset, predicts):
    columns = get_columns(dataset.levels)
    index = get_index(count_train, count_test, dataset)
    return {p.rule: pd.DataFrame(p.confusion_matrix_normalized, index=index, columns=columns) for p in predicts}


def confusion_matrix_multilabel(predicts):
    columns = index = ['Positive', 'Negative']
    return {(idx, p.rule): pd.DataFrame(cfm, index=index, columns=columns)
            for p in predicts
            for idx, cfm in enumerate(p.confusion_matrix_multilabel, start=1)}


def count_train_test(count_train, count_test, dataset) -> pd.DataFrame:
    data = {'labels': [], 'trains': [], 'tests': []}
    for train, test in zip(sorted(count_train.items()), sorted(count_test.items())):
        data['trains'].append(train[1] / dataset.image.patch)
        data['tests'].append(test[1] / dataset.image.patch)
        data['labels'].append(train[0])

    df = pd.DataFrame(data, index=None, columns=list(data.keys()))
    df['labels'] = df[['labels']] \
        .map(lambda row: list(filter(lambda x: x.label.__eq__(row), dataset.levels))[0].specific_epithet)
    return df  #


def evals(predicts) -> pd.DataFrame:
    data = {
        'accuracy': [p.accuracy for p in sorted(predicts, key=lambda x: x.rule)],
        'f1': [p.f1 for p in sorted(predicts, key=lambda x: x.rule)],
        'rule': [p.rule for p in sorted(predicts, key=lambda x: x.rule)]
    }
    return pd.DataFrame(data, index=None, columns=list(data.keys()))


def get_columns(levels):
    """
    Cria uma lista com os levels (classes) que serão utilizados na matriz de confusão.
    :param levels: levels (classes) com nome das espécies utilizadas.
    :return: lista com o nome das classes.
    """
    return [level.specific_epithet for level in levels]


def get_count(count: dict, label: int):
    """
    Encontra a quantidade de treinos de uma determinada classe.
    :param count: coleção com todas as quantidades de treinos.
    :param label: classe que deseja encontrar.
    :return: quantidade de treinos de uma determinada classe.
    """
    for count in list(count.items()):
        if label == count[0]:
            return count[1]


def get_count_train(count, dataset):
    return [int(get_count(count, level.label) / dataset.image.patch)
            for level in sorted(dataset.levels, key=lambda x: x.label)]


def get_count_test(count, dataset):
    return [int(get_count(count, level.label) / dataset.image.patch)
            for level in sorted(dataset.levels, key=lambda x: x.label)]


def get_index(count_train: dict, count_test: dict, dataset: Dataset):
    """
    Cria uma lista com os levels (classes), ou seja, nome e a quantidade de treinos e testes, que serão utilizados na matriz de confusão.
    :param count_train: coleção com a quantidade de treinos da matriz de confusão.
    :param count_test: coleção com a quantidade de testes da matriz de confusão.
    :param levels: levels (classes) com nome das espécies utilizadas.
    :param patch: quantidade de divisões da imagem.
    :return: lista com o nome das classes e a quantidade de treinos de testes.
    """
    return [level.specific_epithet + '(%d-%d)'
            % (int(get_count_train_label(count_train, level.label) / dataset.image.patch),
               int(get_count_test_label(count_test, level.label) / dataset.image.patch))
            for level in sorted(dataset.levels, key=lambda x: x.label)]


def get_count_train_label(count_train: dict, label: int):
    """
    Encontra a quantidade de treinos de uma determinada classe.
    :param count_train: coleção com todas as quantidades de treinos.
    :param label: classe que deseja encontrar.
    :return: quantidade de treinos de uma determinada classe.
    """
    for count in list(count_train.items()):
        if label == count[0]:
            return count[1]


def get_count_test_label(count_test: dict, label: int):
    """
    Encontra a quantidade de testes de uma determinada classe.
    :param count_test: coleção com todas as quantidades de testes.
    :param label: classe que deseja encontrar.
    :return: quantidade de treinos de uma determinada classe.
    """
    for count in list(count_test.items()):
        if label == count[0]:
            return count[1]


def get_level(dataset):
    """
    Cria uma lista com os levels (classes), ou seja, nome e a quantidade de treinos e testes, que serão utilizados na matriz de confusão.
    :param levels: levels (classes) com nome das espécies utilizadas.
    :param patch: quantidade de divisões da imagem.
    :return: lista com o nome das classes e a quantidade de treinos de testes.
    """
    return [level.specific_epithet for level in sorted(dataset.levels, key=lambda x: x.label)]


def infos(final_time, total_test, total_train, total_test_no_patch, total_train_no_patch) -> pd.DataFrame:
    data = {
        'time': [final_time],
        'total_test': [total_test],
        'total_train': [total_train],
        'total_test_no_patch': [total_test_no_patch],
        'total_train_no_patch': [total_train_no_patch],
    }
    return pd.DataFrame(data, index=None, columns=list(data.keys()))


def preds(levels, predicts):
    data = {
        'y_pred+sum': list(itertools.chain(*[p.y_pred.tolist() for p in predicts if p.rule.__eq__('sum')])),
        'y_pred+mult': list(itertools.chain(*[p.y_pred.tolist() for p in predicts if p.rule.__eq__('mult')])),
        'y_pred+max': list(itertools.chain(*[p.y_pred.tolist() for p in predicts if p.rule.__eq__('max')])),
        'y_true': list(itertools.chain(*[predicts[0].y_true.tolist()]))
    }
    df = pd.DataFrame(data, index=None, columns=list(data.keys()))

    if len(levels) > 0:
        df = df.map(lambda row: list(filter(lambda x: x.label.__eq__(row), levels))[0].specific_epithet)

    df['equals'] = df.apply(lambda row: row[row == row['y_true']].index.tolist(), axis=1)
    return df


def set_top(predict, total_test_no_patch):
    data = {
        'k': [topk.k for topk in sorted(predict.topk, key=lambda x: x.k)],
        'topk_accuracy_score': [topk.top_k_accuracy_score for topk in sorted(predict.topk, key=lambda x: x.k)],
        'count_test': np.repeat(total_test_no_patch, len(predict.topk)),
        'topk_accuracy_score+100': [topk.top_k_accuracy_score / total_test_no_patch for topk in
                                    sorted(predict.topk, key=lambda x: x.k)],
        'rule': [predict.rule] * len(predict.topk)  # equivalent a np.repeat, but works in List[str]
    }
    return pd.DataFrame(data, columns=list(data.keys()), index=None)


def tops(predicts, total_test_no_patch):
    data = {'k': [],
            'topk_accuracy_score': [],
            'rule': []}
    df = pd.DataFrame(data, columns=list(data.keys()))
    for predict in predicts:
        top = set_top(predict, total_test_no_patch)
        df = pd.concat([df, top], axis=0)
    return df


def set_true_positive(count_train, count_test, dataset, predict):
    data = {
        'labels': get_level(dataset),
        'count_train': get_count_train(count_train, dataset),
        'count_test': get_count_test(count_test, dataset),
        'true_positive': list(np.diag(predict.confusion_matrix)),
        'rule': [predict.rule] * len(dataset.levels)
    }
    return pd.DataFrame(data, columns=list(data.keys()))


def true_positive(count_train, count_test, dataset, predicts) -> pd.DataFrame:
    data = {
        'labels': [],
        'true_positive': [],
        'rule': []
    }
    df = pd.DataFrame(data, columns=list(data.keys()))
    for predict in predicts:
        best = set_true_positive(count_train, count_test, dataset, predict)
        df = pd.concat([df, best], axis=0)
    return df
