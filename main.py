import click
import datetime
import logging
import numpy as np
import os.path
import sqlalchemy as sa

from database import connect, table_exists
from dataset import Dataset, DataAugmentation
from experiment import Experiment
from model import ResultDB, get_base

datefmt = "%d-%m-%Y+%H-%M-%S"
dateandtime = datetime.datetime.now().strftime(datefmt)
logging.basicConfig(format="\033[32m [%(asctime)s] (%(levelname)s) {%(filename)s %(lineno)d}  %(message)s \033[0m", datefmt="%d/%m/%Y %H:%M:%S", level=logging.INFO)
logging.basicConfig(format="\033[31m [%(asctime)s] (%(levelname)s) {%(filename)s %(lineno)d}  %(message)s \033[0m", datefmt="%d/%m/%Y %H:%M:%S", level=logging.WARNING)
logging.basicConfig(format="\033[35m [%(asctime)s] (%(levelname)s) {%(filename)s %(lineno)d}  %(message)s \033[0m", datefmt="%d/%m/%Y %H:%M:%S", level=logging.CRITICAL)

classifiers = ["DecisionTreeClassifier", "RandomForestClassifier", "KNeighborsClassifier", "MLPClassifier", "SVC"]

def insert_results(dataset, experiment, session):
    f = sa.and_(ResultDB.clf == experiment.classifier.__class__.__name__,
                ResultDB.data_aug == dataset.data_aug,
                ResultDB.input_dir == dataset.input_dir,
                ResultDB.model == dataset.model)
    q = session.query(ResultDB).filter(f).all()
    if len(q) == 0:
        data = [ResultDB(clf=experiment.classifier.__class__.__name__,
                        data_aug = dataset.data_aug,
                        input_dir = dataset.input_dir,
                        model = dataset.model,
                        mean_f1 = float(m.f1),
                        std_f1 = float(m.f1_std),
                        mean_accuracy = float(m.accuracy),
                        std_accuracy = float(m.accuracy_std),
                        rule = m.rule)
               for m in experiment.means]
        session.add_all(data)
        session.commit()
        session.close()


@click.command()
@click.option("-c", "--clf", type=str, required=True, default="DecisionTreeClassifier")
@click.option("-d", "--data_aug", multiple=True, required=False)
@click.option("-i", "--input_dir", required=True)
@click.option("-m", "--min_data_aug", required=False, default=-1)
@click.option("-o", "--output", required=False)
@click.option("-p", "--pca", is_flag=True, default=False)
@click.option("-s", "--sql", is_flag=True, default=False)
def main(clf, data_aug, input_dir, min_data_aug, output, pca, sql):
    # if output is not None and os.path.exists(output):
    #     raise SystemError("output %s already exists" % output)

    if not os.path.exists(input_dir):
        raise SystemExit("input %s not found" % input_dir)

    if clf not in classifiers:
        raise SystemExit("classifier %s not found" % clf)

    data_augmentations = [DataAugmentation(d, min_data_aug) for d in data_aug]
    dataset = Dataset(input_dir)
    dataset.print()
    dataset.load_features()

    experiment = Experiment(clf, data_augmentations, dataset)

    if np.isnan(dataset.x).any():
        raise ValueError("x contains NaN values")

    if pca:
        pass
        # xs = apply_pca(config, dataset, model.features, pca, x)

    experiment.run(output)
    #
    if sql:
        engine, session = connect()

        tables = [ResultDB]
        for t in tables:
            if not table_exists(engine, t.__tablename__):
                base = get_base()
                base.metadata.tables[t.__tablename__].create(bind=engine)
                logging.info("create table: %s" % t.__tablename__)
            else:
                logging.info("table %s already exists" % t.__tablename__)

        insert_results(dataset, experiment, session)

        session.close()
        engine.dispose()

if __name__ == '__main__':
    main()
