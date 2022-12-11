import collections
import csv
import os
import pathlib

import pandas as pd

from save.save_files import save_df


def save_info_samples(fold, list_labels, index_train, index_test, n_patch, path, y, y_train, y_test):
    p = os.path.join(path, str(fold), 'info_samples')
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

    l = []
    for k, v in collections.Counter(y_train).items():
        result_y = {'value': v2 for k2, v2 in collections.Counter(y).items() if k2 == k }
        result_y_test = {'value': v2 for k2, v2 in collections.Counter(y_test).items() if k2 == k }

        percentage_train = round((v * 100) / result_y['value'], 2)
        percentage_test = round((result_y_test['value'] * 100) / result_y['value'], 2)
        samples_train = int(v/n_patch)
        samples_test = int(result_y_test['value']/n_patch)
        samples_total = int(result_y['value']/n_patch)
        labels = list_labels[int(k)-1]['taxon']
        l.append({'label': f'{k} ({labels})', 'samples_train': f'{v} ({samples_train}) ({percentage_train})', 'samples_test': f'{result_y_test["value"]} ({samples_test})({percentage_test})', 'total': f'{result_y["value"]} ({samples_total})'})

    df = pd.DataFrame(l)
    filename = 'samples_fold'
    save_df(df, filename, p, index=False)

    samples_used = {'index_train': [index_train], 'index_test': [index_test]}
    df = pd.DataFrame(samples_used)
    df = df.transpose()
    filename = 'samples_used'
    save_df(df, filename, os.path.join(path, str(fold)), header=False)
