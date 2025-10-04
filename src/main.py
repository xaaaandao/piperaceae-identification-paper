import datetime
import logging
import os

import click

from database.database import connect, create_table, insert_results
from dataset import Dataset, DataAugmentation
from experiment import Experiment
from save import save

datefmt = "%d-%m-%Y+%H-%M-%S"
dateandtime = datetime.datetime.now().strftime(datefmt)
logging.basicConfig(format="\033[32m [%(asctime)s] (%(levelname)s) {%(filename)s %(lineno)d}  %(message)s \033[0m", datefmt="%d/%m/%Y %H:%M:%S", level=logging.INFO)
logging.basicConfig(format="\033[31m [%(asctime)s] (%(levelname)s) {%(filename)s %(lineno)d}  %(message)s \033[0m", datefmt="%d/%m/%Y %H:%M:%S", level=logging.WARNING)
logging.basicConfig(format="\033[35m [%(asctime)s] (%(levelname)s) {%(filename)s %(lineno)d}  %(message)s \033[0m", datefmt="%d/%m/%Y %H:%M:%S", level=logging.CRITICAL)

classifiers = ["DecisionTreeClassifier", "RandomForestClassifier", "KNeighborsClassifier", "MLPClassifier", "SVC"]

@click.command()
@click.option("-c", "--clf", type=str, required=True, default="DecisionTreeClassifier")
@click.option("-d", "--data_aug", multiple=True, required=False)
@click.option("-i", "--input_dir", required=True)
@click.option("-m", "--min_data_aug", required=False, default=-1)
@click.option("-o", "--output_dir", required=False, default="./output")
# @click.option("-p", "--pca", is_flag=True, default=False)
@click.option("-s", "--sql", is_flag=True, default=False)
def main(clf, data_aug, input_dir, min_data_aug, output_dir, sql):
    # if output is not None and os.path.exists(output):
    #     raise SystemError("output %s already exists" % output)

    if not os.path.exists(input_dir):
        raise SystemExit("input %s not found" % input_dir)

    if clf not in classifiers:
        raise SystemExit("classifier %s not found" % clf)

    dataset = Dataset(input_dir)
    data_augmentations = [DataAugmentation(d, min_data_aug) for d in data_aug]

    experiment = Experiment(clf, dataset, data_augmentations)

    experiment.run()

    save(experiment, output_dir)
    if sql:
        engine, session = connect()

        create_table(engine)
        insert_results(experiment, min_data_aug, session)

        session.close()
        engine.dispose()


if __name__ == '__main__':
    main()
