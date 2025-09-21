import click
import datetime
import logging
import os.path

import numpy as np
from sklearn.preprocessing import StandardScaler

from dataset import Dataset
from experiment import Experiment

datefmt = "%d-%m-%Y+%H-%M-%S"
dateandtime = datetime.datetime.now().strftime(datefmt)
logging.basicConfig(format="\033[32m [%(asctime)s] (%(levelname)s) {%(filename)s %(lineno)d}  %(message)s \033[0m", datefmt="%d/%m/%Y %H:%M:%S", level=logging.INFO)
logging.basicConfig(format="\033[31m [%(asctime)s] (%(levelname)s) {%(filename)s %(lineno)d}  %(message)s \033[0m", datefmt="%d/%m/%Y %H:%M:%S", level=logging.WARNING)
logging.basicConfig(format="\033[35m [%(asctime)s] (%(levelname)s) {%(filename)s %(lineno)d}  %(message)s \033[0m", datefmt="%d/%m/%Y %H:%M:%S", level=logging.CRITICAL)

classifiers = ["DecisionTreeClassifier", "RandomForestClassifier", "KNeighborsClassifier", "MLPClassifier", "SVC"]

@click.command()
@click.option("-c", "--clf", type=str, required=True, default="DecisionTreeClassifier")
@click.option("-i", "--input_dir", required=True)
@click.option("-o", "--output", required=False, default="output")
@click.option("-p", "--pca", is_flag=True, default=False)
def main(clf, input_dir, output, pca):
    if not os.path.exists(input_dir):
        raise SystemExit("input %s not found" % input_dir)

    if clf not in classifiers:
        raise SystemExit("classifier %s not found" % clf)

    dataset = Dataset(input_dir)
    dataset.print()
    dataset.load_features()

    experiment = Experiment(clf, dataset, folds=2)

    scaler = StandardScaler()
    dataset.x = scaler.fit_transform(dataset.x)

    if np.isnan(dataset.x).any():
        raise ValueError("x contains NaN values")

    if pca:
        pass
        # xs = apply_pca(config, dataset, model.features, pca, x)

    experiment.run(output)


if __name__ == '__main__':
    main()
