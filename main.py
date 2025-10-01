import datetime
import logging

import click

from dataset import Dataset
from experiment import Experiment

datefmt = "%d-%m-%Y+%H-%M-%S"
dateandtime = datetime.datetime.now().strftime(datefmt)
logging.basicConfig(format="\033[32m [%(asctime)s] (%(levelname)s) {%(filename)s %(lineno)d}  %(message)s \033[0m", datefmt="%d/%m/%Y %H:%M:%S", level=logging.INFO)
logging.basicConfig(format="\033[31m [%(asctime)s] (%(levelname)s) {%(filename)s %(lineno)d}  %(message)s \033[0m", datefmt="%d/%m/%Y %H:%M:%S", level=logging.WARNING)
logging.basicConfig(format="\033[35m [%(asctime)s] (%(levelname)s) {%(filename)s %(lineno)d}  %(message)s \033[0m", datefmt="%d/%m/%Y %H:%M:%S", level=logging.CRITICAL)


@click.command()
@click.option("-c", "--clf", type=str, required=True, default="DecisionTreeClassifier")
# @click.option("-d", "--data_aug", multiple=True, required=False)
@click.option("-i", "--input_dir", required=True)
# @click.option("-m", "--min_data_aug", required=False, default=-1)
@click.option("-o", "--output_dir", required=False)
# @click.option("-p", "--pca", is_flag=True, default=False)
# @click.option("-s", "--sql", is_flag=True, default=False)
def main(clf, input_dir, output_dir):
    dataset = Dataset(input_dir)
    experiment = Experiment(clf, dataset)
    experiment.run()


if __name__ == '__main__':
    main()
