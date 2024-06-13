import os
import pathlib
from typing import LiteralString, Any

import numpy as np
import pandas as pd
import sqlalchemy as sa

from sql.database import insert
from sql.models import Dataset


def exists_dataset(df: pd.DataFrame, session)->Dataset:
    extractor, minimum, n_features, n_samples, name, region = get_dataset_values(df)
    dataset = session.query(Dataset) \
        .filter(sa.and_(Dataset.name.__eq__(name),
                        Dataset.model.__eq__(extractor),
                        Dataset.minimum.__eq__(minimum),
                        Dataset.n_features.__eq__(n_features),
                        Dataset.n_samples.__eq__(n_samples),
                        Dataset.name.__eq__(name),
                        Dataset.region.__eq__(region))) \
        .first()
    return dataset


def create_dataset(**values:dict)->Dataset:
    # extractor, minimum, n_features, n_samples, name, region = get_dataset_values(df)
    return Dataset(values)


def insert_dataset(path: pathlib.Path | LiteralString | str, session)->Dataset:
    filename = os.path.join(path, 'dataset.csv')

    if not os.path.exists(filename):
        print('%s invalid' % filename)
    df = pd.read_csv(filename, sep=';', index_col=False, header=0, na_filter=False)

    dataset = exists_dataset(df, session)

    if not dataset:
        dataset = create_dataset(df)
        insert(dataset, session)

    return dataset

def get_classifier(path: pathlib.Path | LiteralString | str)->Dataset:
    filename = os.path.join(path, 'dataset.csv')
    if not os.path.exists(filename):
        print('%s invalid' % filename)
    df = pd.read_csv(filename, sep=';', index_col=False, header=0, na_filter=False)
    return df['classifier'][0]

def get_dataset_values(df: pd.DataFrame)->Any:
    extractor = get_feature_name(df)
    minimum = int(df['minimum'][0])
    name = df['name'][0]
    n_features = int(df['count_features'][0])
    n_samples = int(df['count_samples'][0])
    region = df['count_features'][0].astype(str)
    return extractor, minimum, n_features, n_samples, name, region


def get_feature_name(df: pd.DataFrame)->str:
    if 'model' in df.columns:
        return df['model'][0]
    if 'descriptor' in df.columns:
        return df['descriptor'][0]
    if 'extractor' in df.columns:
        return df['extractor'][0]


def exists_dataset(session, values:dict) -> Dataset:
    return session.query(Dataset) \
        .filter(sa.and_(Dataset.name.__eq__(values['name']),
                        Dataset.model.__eq__(values['extractor']),
                        Dataset.minimum.__eq__(values['minimum']),
                        Dataset.n_features.__eq__(values['n_features']),
                        Dataset.n_samples.__eq__(values['n_samples']),
                        Dataset.region.__eq__(values['region']))) \
        .first()
