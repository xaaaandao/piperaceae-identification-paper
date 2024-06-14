import os
import pathlib

import pandas as pd

from sql.database import insert, exists_metric
from sql.dataset import insert_dataset
from sql.models import F1, Accuracy, DatasetF1, DatasetAccuracy, TopK, DatasetTopK, Dataset


def loadv2(session):
    for path in pathlib.Path('/home/xandao/v2/').rglob('*ft=*'):
        if len(os.listdir(path)) > 0:
            classifier, dataset = insert_dataset(path, session)
            insert_means(classifier, dataset, path, session)


def insert_means(classifier: str, dataset: Dataset, path: pathlib.Path, session):
    """
    Insere as três métricas em tabelas diferentes, e logo após insere na tabela
    many to many equivalente.
    :param classifier: nome do classificador.
    :param path: caminho do arquivo CSV.
    :return: nada.
    """
    filename = os.path.join(path, 'mean', 'means.csv')
    df = pd.read_csv(filename, sep=';', index_col=False, header=0)
    dict_cols = {j: i for i, j in enumerate(df.columns)}
    for row in df.values:
        if 'f1' in row[dict_cols['metric']]:
            insert_f1(classifier, dataset, dict_cols, row, session)
        if 'accuracy' in row[dict_cols['metric']]:
            insert_accuracy(classifier, dataset, dict_cols, row, session)
        session.commit()

    load_topk(classifier, dataset, path, session)


def load_topk(classifier: str, dataset: Dataset, path: pathlib.Path, session):
    """
    Carrega o arquivo do TopK e insere suas informações no banco de dados.
    :param classifier: nome do classificador.
    :param dataset: classe dataset.
    :param path: caminho do arquivo CSV.
    :param session: sessão do banco de dados.
    :return: nada.
    """
    if exists_metric(classifier, dataset, session, DatasetTopK) > 0:
        return

    filename = os.path.join(path, 'mean', 'means_topk.csv')
    df = pd.read_csv(filename, sep=';', index_col=False, header=0)
    dict_cols = {j: i for i, j in enumerate(df.columns)}
    insert_topk(classifier, dataset, dict_cols, df, session)


def insert_topk(classifier: str, dataset: Dataset, dict_cols:dict, df:pd.DataFrame, session):
    """
    Insere os valores na tabela topk, e logo após insere na tabela many to many
    equivalente.
    :param classifier: nome do classificador.
    :param dataset: classe dataset.
    :param dict_cols: dicionário com as colunas.
    :param df: dataframe do arquivo CSV.
    :param session: sessão do banco de dados.
    :return: nada.
    """
    for row in df.values:
        topk = TopK(k=row[dict_cols['k']],
                    rule=row[dict_cols['rule']],
                    mean=row[dict_cols['mean']],
                    std=row[dict_cols['std']])
        dataset_topk = DatasetTopK(classifier=classifier)
        dataset_topk.topk = topk
        insert(topk, session)
        dataset.topks.append(dataset_topk)
        session.commit()


def insert_accuracy(classifier:str, dataset:Dataset, dict_cols:dict, row, session):
    """
    Insere os valores na tabela acurácia, e logo após insere na tabela many to many
    equivalente.
    :param classifier: nome do classificador.
    :param dataset: classe dataset.
    :param dict_cols: dicionário com as colunas.
    :param row: linha do csv.
    :param session: sessão do banco de dados.
    :return: nada.
    """
    if exists_metric(classifier, dataset, session, DatasetAccuracy) > 0:
        return

    accuracy = Accuracy(rule=row[dict_cols['rule']],
                        mean=row[dict_cols['mean']],
                        std=row[dict_cols['std']])
    dataset_accuracy = DatasetAccuracy(classifier=classifier)
    dataset_accuracy.accuracy = accuracy
    insert(accuracy, session)
    dataset.accuracies.append(dataset_accuracy)


def insert_f1(classifier:str, dataset:Dataset, dict_cols:dict, row, session):
    """
    Insere os valores na tabela f1, e logo após insere na tabela many to many
    equivalente.
    :param classifier: nome do classificador.
    :param dataset: classe dataset.
    :param dict_cols: dicionário com as colunas.
    :param row: linha do csv.
    :param session: sessão do banco de dados.
    :return: nada.
    """
    if exists_metric(classifier, dataset, session, DatasetF1) > 0:
        return

    f1 = F1(mean=row[dict_cols['mean']],
            std=row[dict_cols['std']],
            rule=row[dict_cols['rule']])
    dataset_f1 = DatasetF1(classifier=classifier)
    dataset_f1.f1 = f1
    insert(f1, session)
    dataset.f1s.append(dataset_f1)

