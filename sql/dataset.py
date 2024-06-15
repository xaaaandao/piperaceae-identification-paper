import os
import pathlib
from typing import LiteralString, Any

import numpy as np
import pandas as pd
import sqlalchemy as sa

from sql.database import insert
from sql.models import Dataset, DatasetAccuracy, DatasetF1, DatasetTopK


def create_dataset(**values:dict)->Dataset:
    """
    Retorna uma instância de dataset.
    :param values: dicionário com as informações do dataset.
    :return: dataset
    """
    return Dataset(**values)


def insert_dataset(path: pathlib.Path | LiteralString | str, session)->(str, Dataset):
    """
    Carrega as informações do dataset e adiciona no banco de dados (caso não exista).
    :param path: caminho do arquivo CSV.
    :param session: sessão do banco de dados.
    :return: classifier, dataset
    """
    filename = os.path.join(path, 'dataset.csv')

    if not os.path.exists(filename):
        print('%s invalid' % filename)
    df = pd.read_csv(filename, sep=';', index_col=False, header=0, na_filter=False)
    df.replace(to_replace='None', value=np.nan, inplace=True)
    values = df.to_dict('records')[0]
    classifier = values['classifier']

    df = pd.read_csv(os.path.join(path, 'image.csv'), sep=';', index_col=False, header=0, na_filter=False)
    values_image = df.to_dict('records')[0]

    values.update({'n_features': values['count_features'],'n_samples': values['count_samples'], 'version':2, 'height':values_image['height'], 'width':values_image['width'], 'color': values_image['color'], 'path':path.name})
    for key in ['classifier', 'descriptor', 'extractor', 'format', 'input', 'count_samples', 'count_features']:
        values.__delitem__(key)
    dataset = exists_dataset(session, values)

    if not dataset:
        dataset = create_dataset(**values)
        insert(dataset, session)

    return classifier, dataset


# def get_dataset_values(df: pd.DataFrame)->Any:
#     """
#     Verifica se existe um dataset, e retorna a sua primeira ocorrência.
#     :param session: sessão do banco de dados.
#     :param values: dicionário com as informações do dataset.
#     :return:
#     """
#     extractor = get_feature_name(df)
#     minimum = int(df['minimum'][0])
#     name = df['name'][0]
#     n_features = int(df['count_features'][0])
#     n_samples = int(df['count_samples'][0])
#     region = df['region'][0].astype(str)
#     return extractor, minimum, n_features, n_samples, name, region


def get_feature_name(df: pd.DataFrame)->str:
    """
    Verifica qual coluna está preenchida (model, descriptor ou extractor), e retorna o valor qye está nessa coluna
    :param df: DataFrame com as informações do dataset.
    :return: string com o nome do extrator de features utilizado.
    """
    if 'model' in df.columns:
        return df['model'][0]
    if 'descriptor' in df.columns:
        return df['descriptor'][0]
    if 'extractor' in df.columns:
        return df['extractor'][0]


def exists_dataset(session, values:dict) -> Dataset:
    """
    Verifica se existe um dataset, e retorna a sua primeira ocorrência.
    :param session: sessão do banco de dados.
    :param values: dicionário com as informações do dataset.
    :return:
    """
    return session.query(Dataset) \
        .filter(sa.and_(Dataset.name.__eq__(values['name']),
                        Dataset.count_levels.__eq__(values['count_levels']),
                        Dataset.model.__eq__(values['model']),
                        Dataset.minimum.__eq__(values['minimum']),
                        Dataset.n_features.__eq__(values['n_features']),
                        Dataset.n_samples.__eq__(values['n_samples']),
                        Dataset.region.__eq__(values['region']),
                        Dataset.color.__eq__(values['color']),
                        Dataset.height.__eq__(values['height']),
                        Dataset.path.__eq__(values['path']),
                        Dataset.width.__eq__(values['width']))) \
        .first()


def insert_accuracy(accuracy, classifier, dataset, session):
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
    dataset_accuracy = DatasetAccuracy(classifier=classifier)
    dataset_accuracy.accuracy = accuracy
    insert(accuracy, session)
    dataset.accuracies.append(dataset_accuracy)
    session.commit()


def insert_f1(classifier, dataset, f1, session):
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
    dataset_f1 = DatasetF1(classifier=classifier)
    dataset_f1.f1 = f1
    insert(f1, session)
    dataset.f1s.append(dataset_f1)
    session.commit()


def insert_topk(classifier, dataset, session, topk):
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
    dataset_topk = DatasetTopK(classifier=classifier)
    dataset_topk.topk = topk
    insert(topk, session)
    dataset.topks.append(dataset_topk)
    session.commit()
