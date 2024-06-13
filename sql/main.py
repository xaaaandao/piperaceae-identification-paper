import sqlite3

import numpy as np

from sql.database import connect, close
from sql.models import get_base
from sql.v1 import loadv1, extract_dataset


def main():
    sqlite3.register_adapter(np.int64, lambda val: int(val))
    sqlite3.register_adapter(np.float64, lambda val: float(val))

    engine, session = connect(database='herbario_resultados')
    base = get_base()
    base.metadata.create_all(engine)

    # loadv2(session)
    loadv1(session)
    # test = 'clf=a+len=B+ex=c+ft=d+c=e+dt=f+m=g'
    # extract_dataset(test)

    close(engine, session)


if __name__ == '__main__':
    main()
