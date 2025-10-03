import datetime
import logging
import os

import click

from database import table_exists
from dataset import Dataset
from experiment import Experiment
from model import ResultDB, get_base

datefmt = "%d-%m-%Y+%H-%M-%S"
dateandtime = datetime.datetime.now().strftime(datefmt)
logging.basicConfig(format="\033[32m [%(asctime)s] (%(levelname)s) {%(filename)s %(lineno)d}  %(message)s \033[0m", datefmt="%d/%m/%Y %H:%M:%S", level=logging.INFO)
logging.basicConfig(format="\033[31m [%(asctime)s] (%(levelname)s) {%(filename)s %(lineno)d}  %(message)s \033[0m", datefmt="%d/%m/%Y %H:%M:%S", level=logging.WARNING)
logging.basicConfig(format="\033[35m [%(asctime)s] (%(levelname)s) {%(filename)s %(lineno)d}  %(message)s \033[0m", datefmt="%d/%m/%Y %H:%M:%S", level=logging.CRITICAL)

classifiers = ["DecisionTreeClassifier", "RandomForestClassifier", "KNeighborsClassifier", "MLPClassifier", "SVC"]

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

@click.command()
@click.option("-c", "--clf", type=str, required=True, default="DecisionTreeClassifier")
# @click.option("-d", "--data_aug", multiple=True, required=False)
@click.option("-i", "--input_dir", required=True)
# @click.option("-m", "--min_data_aug", required=False, default=-1)
@click.option("-o", "--output_dir", required=False, default="./output")
# @click.option("-p", "--pca", is_flag=True, default=False)
# @click.option("-s", "--sql", is_flag=True, default=False)
def main(clf, input_dir, output_dir):
    # if output is not None and os.path.exists(output):
    #     raise SystemError("output %s already exists" % output)

    if not os.path.exists(input_dir):
        raise SystemExit("input %s not found" % input_dir)

    if clf not in classifiers:
        raise SystemExit("classifier %s not found" % clf)

    dataset = Dataset(input_dir)

    experiment = Experiment(clf, dataset, folds=2)

    experiment.run(output_dir)

    # if sql:
    #     engine, session = connect()
    #
    #     create_table(engine)
    #     insert_results(experiment, min_data_aug, session)
    #
    #     session.close()
    #     engine.dispose()


if __name__ == '__main__':
    main()
