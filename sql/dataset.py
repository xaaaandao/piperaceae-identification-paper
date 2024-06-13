import os
import pathlib
from typing import LiteralString, Any

import numpy as np
import pandas as pd
import sqlalchemy as sa

from sql.database import insert
from sql.models import Dataset


def create_dataset(**values:dict)->Dataset:
    return Dataset(**values)


def insert_dataset(count_levels: int, path: pathlib.Path | LiteralString | str, session)->(str, Dataset):
    filename = os.path.join(path, 'dataset.csv')

    if not os.path.exists(filename):
        print('%s invalid' % filename)
    df = pd.read_csv(filename, sep=';', index_col=False, header=0, na_filter=False)
    df.replace(to_replace='None', value=np.nan, inplace=True)
    values = df.to_dict('records')[0]
    classifier = values['classifier']

    df = pd.read_csv(os.path.join(path, 'image.csv'), sep=';', index_col=False, header=0, na_filter=False)
    values_image = df.to_dict('records')[0]

    values.update({'n_features': values['count_features'],'n_samples': values['count_samples'], 'version':2, 'height':values_image['height'], 'width':values_image['width'], 'color': values_image['color'], 'path':path.name, 'count_levels': count_levels})
    for key in ['classifier', 'descriptor', 'extractor', 'format', 'input', 'count_samples', 'count_features']:
        values.__delitem__(key)
    dataset = exists_dataset(session, values)

    if not dataset:
        dataset = create_dataset(**values)
        insert(dataset, session)

    return classifier, dataset


def get_dataset_values(df: pd.DataFrame)->Any:
    extractor = get_feature_name(df)
    minimum = int(df['minimum'][0])
    name = df['name'][0]
    n_features = int(df['count_features'][0])
    n_samples = int(df['count_samples'][0])
    region = df['region'][0].astype(str)
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
