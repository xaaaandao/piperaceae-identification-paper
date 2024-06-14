import os
import pathlib
import re
from typing import LiteralString

import pandas as pd

from sql.database import insert, exists_metric
from sql.dataset import exists_dataset, create_dataset
from sql.models import F1, Accuracy, DatasetF1, DatasetAccuracy, TopK, DatasetTopK, Dataset


def loadv1(session):
    for p in pathlib.Path('/media/xandao/6844EF7A44EF4980/resultados/regions').rglob('*clf=*'):
        if len(os.listdir(p)) > 0 and os.path.exists(os.path.join(p, 'info.csv')):
            count_levels = get_count_levels(p)
            classifier, dataset = insert_dataset(count_levels, p, session)
            insert_means(classifier, dataset, p, session)


def get_count_samples(path: pathlib.Path) -> int:
    """
    Retorna a quantidade de amostras.
    :param path: caminho com a localização do arquivo CSV.
    :return: inteiro, com a quantidade de amostras.
    """
    filename = os.path.join(path, 'info.csv')
    df = pd.read_csv(filename, index_col=0, header=None, sep=';')
    return df.loc['n_samples'][1]


def extract_datasetv1(path: pathlib.Path):
    """
    Extrai informações a partir do nome da pasta.
    :param path: caminho com a localização do nome da pasta.
    :return: classifier, values
    """
    if '+r=' in path.name:
        pattern = r'clf=(?P<classifier>.+)\+len=(?P<image_size>.+)\+ex=(?P<model>.+)\+ft=(?P<n_features>.+)\+c=(?P<color>.+)\+dt=(?P<name>.+)\+r=(?P<region>.+)\+m=(?P<minimum>.+)'
    else:
        pattern = r'clf=(?P<classifier>.+)\+len=(?P<image_size>.+)\+ex=(?P<model>.+)\+ft=(?P<n_features>.+)\+c=(?P<color>.+)\+dt=(?P<name>.+)\+m=(?P<minimum>.+)'

    m = re.match(pattern, path.name)
    values = m.groupdict()
    if 'region' not in values.keys():
        values.update({'region': None})
    values.update({'n_samples': get_count_samples(path),
                   'version': 1,
                   'height': values['image_size'],
                   'width': values['image_size'],
                   'path': path.name})
    classifier = values['classifier']
    for key in ['classifier', 'image_size']:
        values.__delitem__(key)
    return classifier, values


def insert_means(classifier:str, dataset:Dataset, path:pathlib.Path, session):
    """
    Carrega o arquivo de cada métrica e da cada regra e insere no banco de dados.
    :param classifier: nome do classificador.
    :param dataset: classe dataset.
    :param path: caminho com a localização do arquivo CSV.
    :param session: sessão do banco de dados.
    :return: nada.
    """
    for rule in ['max', 'mult', 'sum']:
        insert_accuracy(classifier, dataset, path, rule, session)
        insert_f1(classifier, dataset, path, rule, session)
        insert_topk(classifier, dataset, path, rule, session)


def insert_dataset(count_levels:int, path:pathlib.Path, session):
    """
    Extrai as informações do dataset, adiciona-se no banco de dados (caso não exista) e por fim,
    retorna o nome do classificador e o conjunto de dados utilizado.
    :param count_levels: quantidade de espécies.
    :param path: caminho com a localização do arquivo CSV.
    :param session: sessão do banco de dados.
    :return: classifier, dataset
    """
    classifier, values = extract_datasetv1(path)
    values.update({'count_levels': count_levels})
    dataset = exists_dataset(session, values)
    if not dataset:
        dataset = create_dataset(**values)
        insert(dataset, session)
    return classifier, dataset


def get_count_levels(path:pathlib.Path) -> int:
    """
    Retorna a quantidade de espécies.
    :param path: caminho com a localização do arquivo CSV.
    :return: inteiro, com a quantidade de espécies.
    """
    files = [p for p in pathlib.Path(os.path.join(path)).rglob('confusion_matrix+max.csv')]
    df = pd.read_csv(files[0], sep=';', header=0, index_col=0)
    return len(df.index.tolist())


def insert_accuracy(classifier: str, dataset:Dataset, path:pathlib.Path | LiteralString | str, rule: str, session):
    """
    Insere os valores na tabela acurácia, e logo após insere na tabela many to many
    equivalente.
    :param classifier: nome do classificador.
    :param dataset: classe dataset.
    :param path: caminho do arquivo CSV.
    :param rule: regra que foi aplicada no resultado presente daquele CSV.
    :param session: sessão do banco de dados.
    :return: nada.
    """
    if exists_metric(classifier, dataset, session, DatasetAccuracy) > 0:
        return

    filename = os.path.join(path, 'mean', 'accuracy', 'mean+accuracy+%s.csv' % rule)
    df = pd.read_csv(filename, index_col=0, header=None, sep=';')
    accuracy = Accuracy(mean=df.loc['mean_accuracy'][1],
                        std=df.loc['std_accuracy'][1],
                        rule=rule)
    dataset_accuracy = DatasetAccuracy(classifier=classifier)
    dataset_accuracy.accuracy = accuracy
    insert(accuracy, session)
    dataset.accuracies.append(dataset_accuracy)
    session.commit()


def insert_f1(classifier: str, dataset:Dataset, path:pathlib.Path | LiteralString | str, rule: str, session):
    """
    Insere os valores na tabela f1, e logo após insere na tabela many to many
    equivalente.
    :param classifier: nome do classificador.
    :param dataset: classe dataset.
    :param path: caminho do arquivo CSV.
    :param rule: regra que foi aplicada no resultado presente daquele CSV.
    :param session: sessão do banco de dados.
    :return: nada.
    """
    if exists_metric(classifier, dataset, session, DatasetF1) > 0:
        return

    filename = os.path.join(path, 'mean', 'f1', 'mean+f1+%s.csv' % rule)
    df = pd.read_csv(filename, index_col=0, header=None, sep=';')
    f1 = F1(mean=df.loc['mean_f1'][1], std=df.loc['std_f1'][1], rule=rule)
    dataset_f1 = DatasetF1(classifier=classifier)
    dataset_f1.f1 = f1
    insert(f1, session)
    dataset.f1s.append(dataset_f1)
    session.commit()


def insert_topk(classifier: str, dataset:Dataset, path:pathlib.Path | LiteralString | str, rule: str, session):
    """
    Insere os valores na tabela topk, e logo após insere na tabela many to many
    equivalente.
    :param classifier: nome do classificador.
    :param dataset: classe dataset.
    :param path: caminho do arquivo CSV.
    :param rule: regra que foi aplicada no resultado presente daquele CSV.
    :param session: sessão do banco de dados.
    :return: nada.
    """
    if exists_metric(classifier, dataset, session, DatasetTopK) > 0:
        return

    filename = os.path.join(path, 'mean', 'topk', 'mean_topk+%s.csv' % rule)
    df = pd.read_csv(filename, sep=';', index_col=False, header=0)
    dict_cols = {j: i for i, j in enumerate(df.columns)}

    for row in df.values:
        topk = TopK(k=row[dict_cols['k']], mean=row[dict_cols['mean']], std=row[dict_cols['std']], rule=rule)
        dataset_topk = DatasetTopK(classifier=classifier)
        dataset_topk.topk = topk
        insert(topk, session)
        dataset.topks.append(dataset_topk)
        session.commit()
