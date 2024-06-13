import os
import pathlib
import re

import pandas as pd

from sql.database import insert
from sql.dataset import exists_dataset, create_dataset
from sql.models import F1, Accuracy, DatasetF1, DatasetAccuracy, TopK, DatasetTopK


def create_f1(df: pd.DataFrame, rule: str) -> F1:
    return F1(mean_f1=df.loc['mean_f1'][1], std_f1=df.loc['std_f1'][1], rule=rule)


def create_accuracy(df: pd.DataFrame, rule: str) -> Accuracy:
    return Accuracy(mean_accuracy=df.loc['mean_accuracy'][1], std_accuracy=df.loc['std_accuracy'][1], rule=rule)


def get_n_samples(path: pathlib.Path):
    filename = os.path.join(path, 'info.csv')
    df = pd.read_csv(filename, index_col=0, header=None, sep=';')
    return df.loc['n_samples'][1]


def extract_datasetv1(path: pathlib.Path):
    if '+region=' in path.name:
        pattern = r'clf=(?P<classifier>.+)\+len=(?P<image_size>.+)\+ex=(?P<model>.+)\+ft=(?P<n_features>.+)\+c=(?P<color>.+)\+dt=(?P<name>.+)\+r=(?P<region>.+)\+m=(?P<minimum>.+)'
    else:
        pattern = r'clf=(?P<classifier>.+)\+len=(?P<image_size>.+)\+ex=(?P<model>.+)\+ft=(?P<n_features>.+)\+c=(?P<color>.+)\+dt=(?P<name>.+)\+m=(?P<minimum>.+)'

    m = re.match(pattern, path.name)
    values = m.groupdict()
    if 'region' not in values.keys():
        values.update({'region': None})
    values.update(
        {'n_samples': get_n_samples(path), 'version': 1, 'height': values['image_size'], 'width': values['image_size']})
    classifier = values['classifier']
    values.__delitem__('classifier')
    values.__delitem__('image_size')
    return classifier, values


def insert_topk(classifier, dataset, path, rule, session):
    filename = os.path.join(path, 'mean', 'topk', 'mean_topk+%s.csv' % rule)
    df = pd.read_csv(filename, sep=';', index_col=False, header=0)

    dict_cols = {j:i for i,j in enumerate(df.columns)}
    for row in df.values:
        topk = TopK(k=row[dict_cols['k']], mean_score=row[dict_cols['mean']], std_score=row[dict_cols['std']], rule=rule)
        dataset_topk = DatasetTopK(classifier=classifier)
        dataset_topk.topk = topk
        insert(topk, session)
        dataset.topks.append(dataset_topk)
        session.commit()

def loadv1(session):
    for p in pathlib.Path('../output/01-06-2023').rglob('*clf=*'):
        if len(os.listdir(p)) <= 0:
            raise IsADirectoryError('%s invalid' % p.name)

        classifier, values = extract_datasetv1(p)
        dataset = exists_dataset(session, values)
        if not dataset:
            dataset = create_dataset(**values)
            insert(dataset, session)

        for rule in ['max', 'mult', 'sum']:
            insert_accuracy(classifier, dataset, p, rule, session)
            insert_f1(classifier, dataset, p, rule, session)
            insert_topk(classifier, dataset, p, rule, session)



def insert_accuracy(classifier, dataset, path, rule, session):
    filename = os.path.join(path, 'mean', 'accuracy', 'mean+accuracy+%s.csv' % rule)
    df = pd.read_csv(filename, index_col=0, header=None, sep=';')
    accuracy = create_accuracy(df, rule)
    dataset_accuracy = DatasetAccuracy(classifier=classifier)
    dataset_accuracy.accuracy = accuracy
    insert(accuracy, session)
    dataset.accuracies.append(dataset_accuracy)
    session.commit()


def insert_f1(classifier, dataset, path, rule, session):
    filename = os.path.join(path, 'mean', 'f1', 'mean+f1+%s.csv' % rule)
    df = pd.read_csv(filename, index_col=0, header=None, sep=';')
    f1 = create_f1(df, rule)
    dataset_f1 = DatasetF1(classifier=classifier)
    dataset_f1.f1 = f1
    insert(f1, session)
    dataset.f1s.append(dataset_f1)
    session.commit()
