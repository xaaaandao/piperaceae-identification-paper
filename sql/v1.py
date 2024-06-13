import os
import pathlib

import pandas as pd

from sql.models import F1, Accuracy


def create_f1(df: pd.DataFrame, rule: str) -> F1:
    return F1(mean_f1=df.loc['mean_f1'], std_f1=df.loc['std_f1'], rule=rule)


def create_accuracy(df: pd.DataFrame, rule: str) -> Accuracy:
    return Accuracy(mean_accuracy=df.loc['mean_accuracy'], std_accuracy=df.loc['std_accuracy'], rule=rule)


def loadv1(session):
    for p in pathlib.Path('../output/01-06-2023').rglob('*clf=*'):
        if len(os.listdir(p)) <= 0:
            raise IsADirectoryError('%s invalid' % p.name)

        # clf=''
        format = 'clf=%s+ex=%s+%s=%s+%s=%s'

        for rule in ['max', 'mult', 'sum']:
            for metric in ['accuracy', 'topk', 'f1']:
                filename = os.path.join(p, 'mean', metric, 'mean+%s+%s.csv' % (metric, rule))
                df = pd.read_csv(filename, index_col=0, header=None, sep=';')
                accuracy = create_accuracy(df, rule)
                f1 = create_f1(df, rule)
