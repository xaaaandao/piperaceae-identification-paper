from typing import Optional, List

import sqlalchemy
import sqlalchemy.orm

Base = sqlalchemy.orm.declarative_base()


def get_base():
    return Base


class DatasetF1(Base):
    __tablename__ = 'dataset_f1'

    dataset_id: sqlalchemy.orm.Mapped[int] = sqlalchemy.orm.mapped_column(sqlalchemy.ForeignKey('dataset.id'),
                                                                          primary_key=True)
    f1_id: sqlalchemy.orm.Mapped[int] = sqlalchemy.orm.mapped_column(sqlalchemy.ForeignKey('f1.id'), primary_key=True)
    classifier = sqlalchemy.Column(sqlalchemy.String, nullable=True)

    # association between Assocation -> Child
    f1: sqlalchemy.orm.Mapped['F1'] = sqlalchemy.orm.relationship(back_populates='datasets')

    # association between Assocation -> Parent
    dataset: sqlalchemy.orm.Mapped['Dataset'] = sqlalchemy.orm.relationship(back_populates='f1s')


class DatasetAccuracy(Base):
    __tablename__ = 'dataset_accuracy'

    dataset_id: sqlalchemy.orm.Mapped[int] = sqlalchemy.orm.mapped_column(sqlalchemy.ForeignKey('dataset.id'),
                                                                          primary_key=True)
    accuracy_id: sqlalchemy.orm.Mapped[int] = sqlalchemy.orm.mapped_column(sqlalchemy.ForeignKey('accuracy.id'),
                                                                           primary_key=True)
    classifier = sqlalchemy.Column(sqlalchemy.String, nullable=True)

    # association between Assocation -> Child
    accuracy: sqlalchemy.orm.Mapped['Accuracy'] = sqlalchemy.orm.relationship(back_populates='datasets')

    # association between Assocation -> Parent
    dataset: sqlalchemy.orm.Mapped['Dataset'] = sqlalchemy.orm.relationship(back_populates='accuracies')


class DatasetTopK(Base):
    __tablename__ = 'dataset_topk'

    dataset_id: sqlalchemy.orm.Mapped[int] = sqlalchemy.orm.mapped_column(sqlalchemy.ForeignKey('dataset.id'),
                                                                          primary_key=True)
    topk_id: sqlalchemy.orm.Mapped[int] = sqlalchemy.orm.mapped_column(sqlalchemy.ForeignKey('topk.id'),
                                                                           primary_key=True)
    classifier = sqlalchemy.Column(sqlalchemy.String, nullable=True)

    # association between Assocation -> Child
    topk: sqlalchemy.orm.Mapped['TopK'] = sqlalchemy.orm.relationship(back_populates='datasets')

    # association between Assocation -> Parent
    dataset: sqlalchemy.orm.Mapped['Dataset'] = sqlalchemy.orm.relationship(back_populates='topks')

class Dataset(Base):
    __tablename__ = 'dataset'

    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True, autoincrement=True)
    name = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    minimum = sqlalchemy.Column(sqlalchemy.Integer, nullable=True)
    n_features = sqlalchemy.Column(sqlalchemy.Integer, nullable=True)
    n_samples = sqlalchemy.Column(sqlalchemy.Integer, nullable=True)
    width = sqlalchemy.Column(sqlalchemy.Integer, nullable=True)
    height = sqlalchemy.Column(sqlalchemy.Integer, nullable=True)
    version = sqlalchemy.Column(sqlalchemy.Integer, nullable=True)
    color = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    contrast = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    patch = sqlalchemy.Column(sqlalchemy.Integer, nullable=True)
    model = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    region = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    f1s: sqlalchemy.orm.Mapped[List['DatasetF1']] = sqlalchemy.orm.relationship(back_populates='dataset')
    topks: sqlalchemy.orm.Mapped[List['DatasetTopK']] = sqlalchemy.orm.relationship(back_populates='dataset')
    accuracies: sqlalchemy.orm.Mapped[List['DatasetAccuracy']] = sqlalchemy.orm.relationship(back_populates='dataset')


class Accuracy(Base):
    __tablename__ = 'accuracy'

    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True, autoincrement=True)
    accuracy = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    rule = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    datasets: sqlalchemy.orm.Mapped[List['DatasetAccuracy']] = sqlalchemy.orm.relationship(back_populates='accuracy')


class F1(Base):
    __tablename__ = 'f1'

    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True, autoincrement=True)
    f1 = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    rule = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    datasets: sqlalchemy.orm.Mapped[List['DatasetF1']] = sqlalchemy.orm.relationship(back_populates='f1')


class TopK(Base):
    __tablename__ = 'topk'

    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True, autoincrement=True)
    k = sqlalchemy.Column(sqlalchemy.Integer, nullable=True)
    score = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    rule = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    datasets: sqlalchemy.orm.Mapped[List['DatasetTopK']] = sqlalchemy.orm.relationship(back_populates='topk')
