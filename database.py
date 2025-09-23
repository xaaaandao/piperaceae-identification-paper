import os

import sqlalchemy as sa
import sqlalchemy.orm

from model import get_base

def connect(echo=True, user=os.environ["DB_USER"], password=os.environ["DB_PASSWORD"], host=os.environ["DB_HOST"], port=os.environ["DB_PORT"], database=os.environ["DB_NAME"]):
    try:
        url = "postgresql+psycopg2://%s:%s@%s:%s/%s" % (user, password, host, port, database)
        engine = sa.create_engine(url, echo=echo, pool_pre_ping=True)
        session = sqlalchemy.orm.sessionmaker(bind=engine)
        session.configure(bind=engine)
        db = session()
        if engine.connect():
            return engine, db
    except Exception as e:
        raise e


def table_exists(engine, table_name):
    return True if table_name in show_tables(engine) else False


def create_table(engine, table):
    table_name = table.__tablename__

    if not table_exists(engine, table_name):
        base = get_base()
        base.metadata.tables[table_name].create(bind=engine)
        print("create table: %s" % table.__tablename__)


def show_tables(engine):
    return sa.inspect(engine).get_table_names()