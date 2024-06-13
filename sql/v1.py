import os
import pathlib
import re

import pandas as pd

from sql.database import insert
from sql.dataset import exists_dataset, create_dataset
from sql.models import F1, Accuracy, DatasetF1, DatasetAccuracy


def create_f1(df: pd.DataFrame, rule: str) -> F1:
    return F1(mean_f1=df.loc['mean_f1'], std_f1=df.loc['std_f1'], rule=rule)


def create_accuracy(df: pd.DataFrame, rule: str) -> Accuracy:
    return Accuracy(mean_accuracy=df.loc['mean_accuracy'], std_accuracy=df.loc['std_accuracy'], rule=rule)


def extract_dataset(foldername: str) -> dict:
    pattern = r'clf=(?P<classifier>.+)\+len=(?P<image_size>.+)\+ex=(?P<extractor>.+)\+ft=(?P<n_features>.+)\+c=(?P<color>.+)\+dt=(?P<name>.+)\+m=(?P<minimum>.+)'

    m = re.match(pattern, foldername)
    return m.groupdict()


# def exists_dataset(foldername: str) -> bool:

def loadv1(session):
    for p in pathlib.Path('../output/01-06-2023').rglob('*clf=*'):
        if len(os.listdir(p)) <= 0:
            raise IsADirectoryError('%s invalid' % p.name)

        values = extract_dataset(p.name)
        dataset = exists_dataset(session, values)
        if not dataset:
            dataset = create_dataset(**values)
            insert(dataset, session)

        for rule in ['max', 'mult', 'sum']:
            insert_accuracy(dataset, p, rule, session, values)
            insert_f1(dataset, p, rule, session, values)
            session.commit()


def insert_accuracy(dataset, path, rule, session, values):
    filename = os.path.join(path, 'mean', 'accuracy', 'mean+accuracy+%s.csv' % rule)
    df = pd.read_csv(filename, index_col=0, header=None, sep=';')
    accuracy = create_accuracy(df, rule)
    dataset_accuracy = DatasetAccuracy(classifier=values['classifier'])
    dataset_accuracy.accuracy = accuracy
    insert(accuracy, session)
    dataset.accuracies.append(dataset_accuracy)


def insert_f1(dataset, path, rule, session, values):
    filename = os.path.join(path, 'mean', 'f1', 'mean+f1+%s.csv' % rule)
    df = pd.read_csv(filename, index_col=0, header=None, sep=';')
    f1 = create_f1(df, rule)
    dataset_f1 = DatasetF1(classifier=values['classifier'])
    dataset_f1.f1 = f1
    insert(f1, session)
    dataset.f1s.append(dataset_f1)
