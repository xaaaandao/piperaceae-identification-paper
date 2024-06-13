import os
import pathlib
from typing import LiteralString, Any

import pandas as pd
import sqlalchemy

from sql.database import insert
from sql.dataset import insert_dataset
from sql.models import F1, Accuracy, DatasetF1, DatasetAccuracy, TopK, DatasetTopK, Dataset


def loadv2(session):
    for p in pathlib.Path('../output/pr_dataset/a').glob('*'):
        if len(os.listdir(p)) > 0:
            classifier, dataset = insert_dataset(p, session)


            filename = os.path.join(p, 'mean', 'means.csv')
            df = pd.read_csv(filename, sep=';', index_col=False, header=0)

            dict_cols = {j:i for i,j in enumerate(df.columns)}

            for row in df.values:
                if 'f1' in row[dict_cols['metric']]:
                    c = session.query(DatasetF1).filter(sqlalchemy.and_(DatasetF1.dataset.__eq__(dataset), DatasetF1.classifier.__eq__(classifier))).count()
                    if c == 0:
                        insert_f1(classifier, dataset, dict_cols, row, session)
                if 'accuracy' in row[dict_cols['metric']]:
                    c = session.query(DatasetAccuracy).filter(sqlalchemy.and_(DatasetAccuracy.dataset.__eq__(dataset), DatasetAccuracy.classifier.__eq__(classifier))).count()
                    if c==0:
                        insert_accuracy(classifier, dataset, dict_cols, row, session)

                session.commit()

            insert_topk(classifier, dataset, p, session)


def insert_topk(classifier:str, dataset:Dataset, p, session):
    filename = os.path.join(p, 'mean', 'means_topk.csv')
    df = pd.read_csv(filename, sep=';', index_col=False, header=0)
    dict_cols = {j: i for i, j in enumerate(df.columns)}
    c = session.query(DatasetTopK).filter(sqlalchemy.and_(DatasetTopK.dataset.__eq__(dataset), DatasetTopK.classifier.__eq__(classifier))).count()
    if c == 0:
        for row in df.values:
            topk = TopK(k=row[dict_cols['k']], rule=row[dict_cols['rule']], mean=row[dict_cols['mean']], std=row[dict_cols['std']])
            dataset_topk = DatasetTopK(classifier=classifier)
            dataset_topk.topk = topk
            insert(topk, session)
            dataset.topks.append(dataset_topk)

            session.commit()


def insert_accuracy(classifier, dataset, dict_cols, row, session):
    accuracy = Accuracy(rule=row[dict_cols['rule']], mean=row[dict_cols['mean']], std=row[dict_cols['std']])
    dataset_accuracy = DatasetAccuracy(classifier=classifier)
    dataset_accuracy.accuracy = accuracy
    insert(accuracy, session)
    dataset.accuracies.append(dataset_accuracy)


def insert_f1(classifier, dataset, dict_cols, row, session):
    f1 = F1(mean=row[dict_cols['mean']], std=row[dict_cols['std']], rule=row[dict_cols['rule']])
    dataset_f1 = DatasetF1(classifier=classifier)
    dataset_f1.f1 = f1
    insert(f1, session)
    dataset.f1s.append(dataset_f1)


def get_image(filename):
    df = pd.read_csv(filename, sep=';', index_col=False, header=0)
    constrast = df['constrast'][0]
    height = df['height'][0]
    width = df['width'][0]
    patch = df['patch'][0]
    color = df['color'][0]

    return color, height, width, patch
