import os
import pathlib
import sqlite3

import numpy as np
import pandas as pd
import sqlalchemy as sa

from sql.database import connect, close
from sql.models import get_base
from sql.v2 import loadv2


def loadv1(session):
    for p in pathlib.Path('../output/01-06-2023').rglob('*clf=*'):
        if len(os.listdir(p)) <= 0:
            raise IsADirectoryError('%s invalid' % p.name)


        for rule in ['max', 'mult', 'sum']:
            for metric in ['accuracy']:
                filename = os.path.join(p, 'mean', 'accuracy', 'mean+%s+%s.csv' % (metric, rule))
                print(filename)
                df = pd.read_csv(filename, index_col=0, header=None, sep=';')
                print(df.loc['mean_accuracy'][1])
                print(df.loc['mean_accuracy'][1])




def main():
    # sqlite3.register_adapter(np.int64, lambda val: int(val))
    # sqlite3.register_adapter(np.float64, lambda val: float(val))
    #
    # engine, session = connect(database='herbario_resultados')
    # base = get_base()
    # base.metadata.create_all(engine)

    # loadv2(session)
    loadv1(None)

    # close(engine, session)


if __name__ == '__main__':
    main()
