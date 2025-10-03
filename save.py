import logging

import pandas as pd


def save_csv(df: pd.DataFrame, filename: str, header=True, index=False):
    df.to_csv(filename, sep=";", quoting=2, index=index, header=header, encoding="utf-8")
    logging.info("saving %s" % filename)


def save_csv_transpose(data, filename, header, index):
    df = pd.DataFrame(data, columns=list(data.keys()))
    df = df.transpose()
    save_csv(df, filename, header=header, index=index)
