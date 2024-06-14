import os
from typing import Any

import sqlalchemy
import sqlalchemy as sa
import sqlalchemy.ext.declarative
import sqlalchemy.orm
import sqlalchemy.schema

from sql.models import Dataset


def connect(echo: bool = True,
            host: str = 'localhost',
            user: str = os.environ['PGUSER'],
            password: str = os.environ['PGPWD'],
            port: str = '5432',
            database: str = 'herbario'):
    """
    Estabelece uma conexão com o banco de dados.
    :param echo:
    :param host: ip do banco de dados.
    :param user: usuário do banco de dados.
    :param password: senha do usuário do banco de dados.
    :param port: porta do banco de dados.
    :param database: nome do banco de dados.
    :return: engine, session
    """
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



def insert(data: Any, session):
    """
    Insere um registro no banco de dados.
    :param data: registro que será inserido.
    :param session: sessão do banco de dados.
    :return: nada.
    """
    try:
        session.add(data)
        session.commit()
    except Exception as e:
        print(e)
        session.rollback()



def inserts(datas: list, session):
    """
    Insere mais de um registro no banco de dados.
    :param datas: conjunto de registros que será inserido.
    :param session: sessão do banco de dados.
    :return: nada.
    """
    try:
        session.add_all(datas)
        session.commit()
    except Exception as e:
        print(e)
        session.rollback()



def close(engine, session):
    """
    Encerra a conexão com o banco de dados.
    :param engine:
    :param session: sessão do banco de dados.
    :return: nada.
    """
    engine.dispose()
    session.close()


def exists_metric(classifier: str, dataset:Dataset, session, table: Any):
    """
    Retorna a quantidade de registros de um determinado dataset e classificador.
    :param classifier: nome do classificador que está sendo procurado.
    :param dataset: classe dataset.
    :param session: sessão do banco de dados.
    :param table: tabela aonde deve ser feito a consulta.
    :return: quantidade de registros.
    """
    return session.query(table) \
        .filter(sqlalchemy.and_(table.dataset.__eq__(dataset),
                                table.classifier.__eq__(classifier))) \
        .count()
