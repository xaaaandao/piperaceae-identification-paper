import os
import pathlib
import re
from typing import LiteralString

import pandas as pd
import sqlalchemy

from sql.database import insert
from sql.dataset import exists_dataset, create_dataset
from sql.models import F1, Accuracy, DatasetF1, DatasetAccuracy, TopK, DatasetTopK, Dataset


def get_n_samples(path: pathlib.Path):
    filename = os.path.join(path, 'info.csv')
    df = pd.read_csv(filename, index_col=0, header=None, sep=';')
    return df.loc['n_samples'][1]


def extract_datasetv1(path: pathlib.Path):
    if '+r=' in path.name:
        pattern = r'clf=(?P<classifier>.+)\+len=(?P<image_size>.+)\+ex=(?P<model>.+)\+ft=(?P<n_features>.+)\+c=(?P<color>.+)\+dt=(?P<name>.+)\+r=(?P<region>.+)\+m=(?P<minimum>.+)'
    else:
        pattern = r'clf=(?P<classifier>.+)\+len=(?P<image_size>.+)\+ex=(?P<model>.+)\+ft=(?P<n_features>.+)\+c=(?P<color>.+)\+dt=(?P<name>.+)\+m=(?P<minimum>.+)'

    m = re.match(pattern, path.name)
    values = m.groupdict()
    if 'region' not in values.keys():
        values.update({'region': None})
    values.update(
        {'n_samples': get_n_samples(path), 'version': 1, 'height': values['image_size'], 'width': values['image_size'], 'path': path.name})
    classifier = values['classifier']
    values.__delitem__('classifier')
    values.__delitem__('image_size')
    return classifier, values


def loadv1(session):
    for p in pathlib.Path('/media/xandao/6844EF7A44EF4980/resultados/regions').rglob('*clf=*'):
        if len(os.listdir(p)) > 0 and os.path.exists(os.path.join(p, 'info.csv')):
            files = [p for p in pathlib.Path(os.path.join(p)).rglob('confusion_matrix+max.csv')]
            df = pd.read_csv(files[0], sep=';', header=0, index_col=0)
            count_levels = len(df.index.tolist())

            classifier, values = extract_datasetv1(p)
            values.update({'count_levels': count_levels})
            dataset = exists_dataset(session, values)
            if not dataset:
                dataset = create_dataset(**values)
                insert(dataset, session)

            for rule in ['max', 'mult', 'sum']:
                insert_accuracy(classifier, dataset, p, rule, session)
                insert_f1(classifier, dataset, p, rule, session)
                insert_topk(classifier, dataset, p, rule, session)
            # break


def insert_accuracy(classifier: str, dataset:Dataset, path:pathlib.Path | LiteralString | str, rule: str, session):
    filename = os.path.join(path, 'mean', 'accuracy', 'mean+accuracy+%s.csv' % rule)
    c = session.query(DatasetAccuracy).filter(sqlalchemy.and_(DatasetAccuracy.dataset.__eq__(dataset), DatasetAccuracy.classifier.__eq__(classifier))).count()
    if c == 0:
        df = pd.read_csv(filename, index_col=0, header=None, sep=';')
        accuracy = Accuracy(mean=df.loc['mean_accuracy'][1], std=df.loc['std_accuracy'][1], rule=rule)
        dataset_accuracy = DatasetAccuracy(classifier=classifier)
        dataset_accuracy.accuracy = accuracy
        insert(accuracy, session)
        dataset.accuracies.append(dataset_accuracy)
        session.commit()


def insert_f1(classifier: str, dataset:Dataset, path:pathlib.Path | LiteralString | str, rule: str, session):
    c = session.query(DatasetF1).filter(sqlalchemy.and_(DatasetF1.dataset.__eq__(dataset), DatasetF1.classifier.__eq__(classifier))).count()
    if c == 0:
        filename = os.path.join(path, 'mean', 'f1', 'mean+f1+%s.csv' % rule)
        df = pd.read_csv(filename, index_col=0, header=None, sep=';')
        f1 = F1(mean=df.loc['mean_f1'][1], std=df.loc['std_f1'][1], rule=rule)
        dataset_f1 = DatasetF1(classifier=classifier)
        dataset_f1.f1 = f1
        insert(f1, session)
        dataset.f1s.append(dataset_f1)
        session.commit()


def insert_topk(classifier: str, dataset:Dataset, path:pathlib.Path | LiteralString | str, rule: str, session):
    filename = os.path.join(path, 'mean', 'topk', 'mean_topk+%s.csv' % rule)
    df = pd.read_csv(filename, sep=';', index_col=False, header=0)

    dict_cols = {j: i for i, j in enumerate(df.columns)}
    c = session.query(DatasetTopK).filter(sqlalchemy.and_(DatasetTopK.dataset.__eq__(dataset), DatasetTopK.classifier.__eq__(classifier))).count()
    if c == 0:
        for row in df.values:
            topk = TopK(k=row[dict_cols['k']], mean=row[dict_cols['mean']], std=row[dict_cols['std']],
                        rule=rule)
            dataset_topk = DatasetTopK(classifier=classifier)
            dataset_topk.topk = topk
            insert(topk, session)
            dataset.topks.append(dataset_topk)
            session.commit()
