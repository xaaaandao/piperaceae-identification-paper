import logging
import os

import sqlalchemy as sa
import sqlalchemy.orm

from database.model import ResultDB, get_base


def connect(echo=True, user=os.environ["DB_USER"], password=os.environ["DB_PASSWORD"], host="127.0.0.1", port="5432", database="herbario"):
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


def insert_results(experiment, min_data_aug, session):
    data_augmentations = [d.input_dir for d in experiment.data_augmentations]
    data = [ResultDB(clf=experiment.classifier.__class__.__name__,
                     data_aug=str(data_augmentations),
                     input_dir=experiment.dataset.input_dir,
                     model=experiment.dataset.model,
                     min_data_aug = int(min_data_aug),
                     mean_f1=float(m.f1),
                     std_f1=float(m.f1_std),
                     mean_accuracy=float(m.accuracy),
                     std_accuracy=float(m.accuracy_std),
                     rule=m.rule)
            for m in experiment.means]
    session.add_all(data)
    session.commit()
    session.close()


def create_table(engine):
    tables = [ResultDB]
    for t in tables:

        if not table_exists(engine, t.__tablename__):
            base = get_base()
            base.metadata.tables[t.__tablename__].create(bind=engine)
            logging.info("create table: %s" % t.__tablename__)
        else:
            logging.info("table %s already exists" % t.__tablename__)


def show_tables(engine):
    return sa.inspect(engine).get_table_names()