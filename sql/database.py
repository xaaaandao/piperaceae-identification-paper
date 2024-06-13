import os
from typing import Any

import sqlalchemy
import sqlalchemy as sa
import sqlalchemy.ext.declarative
import sqlalchemy.orm
import sqlalchemy.schema


def connect(echo: bool = True, host: str = 'localhost', user: str = os.environ['PGUSER'],
            password: str = os.environ['PGPWD'], port: str = '5432', database: str = 'herbario'):
    try:
        url = sa.URL.create(
            'postgresql+psycopg2',
            username=user,
            password=password,
            host=host,
            database=database,
            port=port
        )
        engine = sa.create_engine(url, echo=echo, pool_pre_ping=True)
        session = sqlalchemy.orm.sessionmaker(bind=engine)
        session.configure(bind=engine)
        if engine.connect():
            return engine, session()
    except Exception as e:
        print(e)


def table_is_empty(query):
    return query == 0


def insert(data: Any, session):
    try:
        session.add(data)
        session.commit()
    except Exception as e:
        print(e)
        session.rollback()



def inserts(data: list, session):
    try:
        session.add_all(data)
        session.commit()
    except Exception as e:
        print(e)
        session.rollback()

def update(clause, session, table, values):
    try:
        session.query(table)\
            .filter(clause)\
            .update(values=values, synchronize_session=False)
        session.commit()
    except Exception as e:
        print(e)
        session.rollback()


def close(engine, session):
    engine.dispose()
    session.close()
