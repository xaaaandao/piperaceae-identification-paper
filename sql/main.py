import sqlite3

import numpy as np

from sql.database import connect, close
from sql.models import get_base
from sql.v1 import loadv1, extract_datasetv1
from sql.v2 import loadv2


def main():
    sqlite3.register_adapter(np.int64, lambda val: int(val))
    sqlite3.register_adapter(np.float64, lambda val: float(val))

    engine, session = connect(database='herbario-results')
    base = get_base()
    base.metadata.create_all(engine)

    loadv1(session)
    # loadv2(session)

    close(engine, session)


if __name__ == '__main__':
    main()
