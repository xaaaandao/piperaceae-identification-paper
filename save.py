import pathlib
from typing import LiteralString, Any

import joblib
import logging
import os
import pandas as pd

from config import Config
from dataset import Dataset


def save_best(df: pd.DataFrame, output: pathlib.Path | LiteralString | str):
    """
    Salva em um arquivo CSV a melhor média, ou seja, a médiaa com o maior valor
    de f1 e acurácia.
    :param df: Dataframe aonde será procurado o melhor valor.
    :param output: local aonde as informações do melhor classificador deve ser salvo.
    """
    filename = os.path.join(output, 'best+means.csv')
    data = {
        'mean': [df.query('metric == \'f1\'').query('mean == mean.max()')['mean'].values[0],
                 df.query('metric == \'accuracy\'').query('mean == mean.max()')['mean'].values[0]],
        'std': [df.query('metric == \'f1\'').query('mean == mean.max()')['std'].values[0],
                df.query('metric == \'accuracy\'').query('mean == mean.max()')['std'].values[0]],
        'metric': ['f1', 'accuracy'],
        'rule': [df.query('metric == \'f1\'').query('mean == mean.max()')['rule'].values[0],
                 df.query('metric == \'accuracy\'').query('mean == mean.max()')['rule'].values[0]]
    }
    df = pd.DataFrame(data, columns=data.keys())
    df.to_csv(filename, index=False, header=True, sep=';', quoting=2, encoding='utf-8')
    logging.info('Saving %s' % filename)


def save_means(levels: list, means: list, output: pathlib.Path | LiteralString | str):
    """
    Salva em um arquivo CSV todas as médias (topk, acurácia, f1, tempo).
    :param levels: lista com todas as espécies presentes no experimentos.
    :param means: lista com todas as médias das execuções.
    :param output: local aonde as informações do melhor classificador deve ser salvo.
    """
    output = os.path.join(output, 'mean')
    os.makedirs(output, exist_ok=True)

    data = {'mean': [], 'metric': [], 'std': [], 'rule': []}
    df = pd.DataFrame(data, columns=data.keys())

    data = {'k': [], 'mean': [], 'std': [], 'rule': []}
    df_topk = pd.DataFrame(data, columns=data.keys())

    data = {'label': [], 'mean': [], 'std': [], 'rule': []}
    df_true_positive = pd.DataFrame(data, columns=data.keys())
    for mean in means:
        df = pd.concat([df, mean.save()], axis=0)
        df_topk = pd.concat([df_topk, mean.save_topk()], axis=0)
        df_true_positive = pd.concat([df_true_positive, mean.save_true_positive(levels)], axis=0)

    filename = os.path.join(output, 'means.csv')
    df.to_csv(filename, index=False, header=True, sep=';', quoting=2, encoding='utf-8')
    logging.info('Saving %s' % filename)

    filename = os.path.join(output, 'means_topk.csv')
    df_topk.to_csv(filename, index=False, header=True, sep=';', quoting=2, encoding='utf-8')
    logging.info('Saving %s' % filename)

    filename = os.path.join(output, 'means_true_positive.csv')
    df_true_positive.to_csv(filename, index=False, header=True, sep=';', quoting=2, encoding='utf-8')
    logging.info('Saving %s' % filename)

    save_best(df, output)


def find_fold_rule(attr:str, best:object, folds:list):
    """
    Encontra o fold que possui o melhor resultado (depende da métrica).
    :param attr: nome do atributo (f1 ou accuracy).
    :param best: o fold com o melhor fold.
    :param folds: lista com todas as execuções dos folds.
    :return: fold, predict
    """
    try:
        for fold in folds:
            for predict in fold.result.predicts:
                if getattr(predict.eval, attr) == best:
                    return fold, predict
    except:
        raise ValueError


def save_best_fold(folds: list, output: pathlib.Path | LiteralString | str):
    """
    Salva em um arquivo CSV o fold com o maior valor de f1 e o maior valor de acurácia.
    :param folds: lista com todas as execuções dos folds.
    :param output: local aonde as informações do melhor classificador deve ser salvo.
    """
    output = os.path.join(output, 'best')
    os.makedirs(output, exist_ok=True)
    filename = os.path.join(output, 'best_fold.csv')

    best_f1 = max(predict.eval.f1 for fold in folds for predict in fold.result.predicts)
    best_accuracy = max(predict.eval.accuracy for fold in folds for predict in fold.result.predicts)
    fold_f1, rule_f1 = find_fold_rule( 'f1', best_f1, folds)
    fold_accuracy, rule_accuracy = find_fold_rule( 'accuracy', best_accuracy, folds)
    data = {'metric': [best_f1, best_accuracy],
            'fold': [fold_f1.fold, fold_accuracy.fold],
            'rule': [rule_f1.rule, rule_accuracy.rule]}
    df = pd.DataFrame(data, columns=data.keys())
    df.to_csv(filename, index=False, header=True, sep=';', quoting=2, encoding='utf-8')
    logging.info('Saving %s' % filename)


def save_folds(dataset: Dataset, folds: list, output: pathlib.Path | LiteralString | str):
    """
    Salva em um arquivo CSV informações de cada fold executado no experimento.
    :param dataset: classe dataset com informações do conjunto de dados.
    :param folds: lista com todas as execuções dos folds.
    :param output: local aonde as informações do melhor classificador deve ser salvo.
    """
    for fold in folds:
        fold.save(dataset.levels, output, dataset.image.patch)
    save_best_fold(folds, output)


def save_best_classifier(classifier: Any, output: pathlib.Path | LiteralString | str):
    """
    Salva o melhor classificador encontrado pela função GridSearchCV.
    :param classifier: classificador com os melhores hiperparâmetros.
    :param output: local aonde o melhor classificador deve ser salvo.
    """
    output = os.path.join(output, 'best')
    os.makedirs(output, exist_ok=True)
    filename = os.path.join(output, 'best_classifier.pkl')
    logging.info('[CLASSIFIER] Saving %s' % filename)

    try:
        with open(filename, 'wb') as file:
            joblib.dump(classifier, file, compress=3)
        file.close()
    except FileExistsError:
        logging.warning('problems in save model (%s)' % filename)


def save_best_info_classifier(classifier: Any, output: pathlib.Path | LiteralString | str):
    """
    Salva em um arquivo CSV as informações com o melhor classificador encontrado pela função GridSearchCV.
    :param classifier: classificador com os melhores hiperparâmetros.
    :param output: local aonde as informações do melhor classificador deve ser salvo.
    """
    output = os.path.join(output, 'best')
    os.makedirs(output, exist_ok=True)
    filename = os.path.join(output, 'best_classifier.csv')

    df = pd.DataFrame(classifier.cv_results_)
    df.to_csv(filename, index=False, header=True, sep=';', quoting=2, encoding='utf-8')
    logging.info('Saving %s' % filename)


def save(classifier: Any, config: Config, dataset: Dataset, folds: list, means: list,
         output: pathlib.Path | LiteralString | str):
    """
    Chama todas as funções que salvam.
    :param classifier: classificador com os melhores hiperparâmetros.
    :param config: classe config com os valores das configurações dos experimentos.
    :param dataset: classe dataset com informações do conjunto de dados.
    :param folds: lista com todas as execuções dos folds.
    :param means: lista com todas as médias das execuções.
    :param output: local aonde as informações do melhor classificador deve ser salvo.
    """
    config.save(output)
    dataset.save(classifier, output)

    save_best_classifier(classifier, output)
    save_best_info_classifier(classifier, output)
    save_folds(dataset, folds, output)
    save_means(dataset.levels, means, output)