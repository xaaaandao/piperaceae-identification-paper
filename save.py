import pathlib
from typing import LiteralString, Any

import joblib
import logging
import os
import pandas as pd

from config import Config
from dataset import Dataset


def save_best_fold(folds: list, output: pathlib.Path | LiteralString | str):
    """
    Salva em um arquivo CSV o fold com o maior valor de f1 e o maior valor de acurácia.
    :param folds: lista com todas as execuções dos folds.
    :param output: local aonde as informações do melhor classificador deve ser salvo.
    """
    filename = os.path.join(output, 'best_fold.csv')

    data = {'accuracy': [],
            'f1': [],
            'rule': [],
            'fold': []}
    best = pd.DataFrame(data, columns=list(data.keys()))
    for fold in folds:
        eval = fold.dataframes['evals']
        eval['fold'] = fold.fold
        best = pd.concat([best, eval], axis=0)
    best_accuracy = best.loc[best['accuracy'] == best['accuracy'].max()]
    best_f1 = best.loc[best['f1'] == best['f1'].max()]
    data = {'metric': ['accuracy', 'f1'],
            'value': [best_accuracy['accuracy'].values[0], best_f1['f1'].values[0]],
            'rule': [best_accuracy['rule'].values[0], best_f1['rule'].values[0]],
            'fold': [best_accuracy['fold'].values[0], best_f1['fold'].values[0]]}
    df = pd.DataFrame(data, columns=list(data.keys()))
    df.to_csv(filename, index=False, header=True, sep=';', quoting=2, encoding='utf-8')
    logging.info('Saving %s' % filename)



def save_best_classifier(classifier: Any, output: pathlib.Path | LiteralString | str):
    """
    Salva o melhor classificador encontrado pela função GridSearchCV.
    :param classifier: classificador com os melhores hiperparâmetros.
    :param output: local aonde o melhor classificador deve ser salvo.
    """
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
    filename = os.path.join(output, 'best_classifier.csv')

    df = pd.DataFrame(classifier.cv_results_)
    df.to_csv(filename, index=False, header=True, sep=';', quoting=2, encoding='utf-8')
    logging.info('Saving %s' % filename)


def save(classifier: Any, config: Config, dataset: Dataset, folds: list,
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
    save_best(classifier, folds, output)


def save_best(classifier, folds, output):
    output = os.path.join(output, 'best')
    os.makedirs(output, exist_ok=True)
    save_best_classifier(classifier, output)
    save_best_info_classifier(classifier, output)
    save_best_fold(folds, output)
