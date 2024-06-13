import os
import pathlib
from typing import LiteralString, Any

import pandas as pd

from sql.database import insert
from sql.dataset import insert_dataset, get_classifier
from sql.models import F1, Accuracy, DatasetF1, DatasetAccuracy, TopK, DatasetTopK




def loadv2(session):
    for p in pathlib.Path('../output/pr_dataset/a').glob('*'):
        if len(os.listdir(p)) <= 0:
            print('%s invalid' % p.name)
        dataset = insert_dataset(p, session)
        classifier = get_classifier(p)


        filename = os.path.join(p, 'mean', 'means.csv')
        df = pd.read_csv(filename, sep=';', index_col=False, header=0)

        dict_cols = {j:i for i,j in enumerate(df.columns)}

        for row in df.values:
            if 'f1' in row[dict_cols['metric']]:
                f1 = F1(f1=row[dict_cols['mean']],rule=row[dict_cols['rule']])
                dataset_f1 = DatasetF1(classifier=classifier)
                dataset_f1.f1 = f1
                insert(f1, session)
                dataset.f1s.append(dataset_f1)
            if 'accuracy' in row[dict_cols['metric']]:
                accuracy = Accuracy(rule=row[dict_cols['rule']],accuracy=row[dict_cols['mean']])
                dataset_accuracy = DatasetAccuracy(classifier=classifier)
                dataset_accuracy.accuracy = accuracy
                insert(accuracy, session)
                dataset.accuracies.append(dataset_accuracy)

            session.commit()

        filename = os.path.join(p, 'mean', 'means_topk.csv')
        df = pd.read_csv(filename, sep=';', index_col=False, header=0)

        dict_cols = {j:i for i,j in enumerate(df.columns)}

        for row in df.values:

            topk = TopK(k=row[dict_cols['k']], rule=row[dict_cols['rule']], score=row[dict_cols['score']])
            dataset_topk = DatasetTopK(classifier=classifier)
            dataset_topk.topk = topk
            insert(topk, session)
            dataset.topks.append(dataset_topk)

            session.commit()

def get_image(filename):
    df = pd.read_csv(filename, sep=';', index_col=False, header=0)
    constrast = df['constrast'][0]
    height = df['height'][0]
    width = df['width'][0]
    patch = df['patch'][0]
    color = df['color'][0]

    return color, height, width, patch
