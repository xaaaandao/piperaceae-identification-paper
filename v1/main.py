import click
import datetime
import logging
import numpy as np
import os.path

from database import connect
from dataset import Dataset, DataAugmentation
from experiment import Experiment

datefmt = "%d-%m-%Y+%H-%M-%S"
dateandtime = datetime.datetime.now().strftime(datefmt)
logging.basicConfig(format="\033[32m [%(asctime)s] (%(levelname)s) {%(filename)s %(lineno)d}  %(message)s \033[0m", datefmt="%d/%m/%Y %H:%M:%S", level=logging.INFO)
logging.basicConfig(format="\033[31m [%(asctime)s] (%(levelname)s) {%(filename)s %(lineno)d}  %(message)s \033[0m", datefmt="%d/%m/%Y %H:%M:%S", level=logging.WARNING)
logging.basicConfig(format="\033[35m [%(asctime)s] (%(levelname)s) {%(filename)s %(lineno)d}  %(message)s \033[0m", datefmt="%d/%m/%Y %H:%M:%S", level=logging.CRITICAL)




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
    if sql:
        engine, session = connect()

        create_table(engine)
        insert_results(experiment, min_data_aug, session)

        session.close()
        engine.dispose()


if __name__ == '__main__':
    main()
