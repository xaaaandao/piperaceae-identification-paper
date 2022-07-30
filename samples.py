import dataclasses

import numpy
import sklearn.model_selection


def get_samples_with_patch(x, y, list_index, n_patch):
    new_x = numpy.zeros(shape=(0, x.shape[1]))
    new_y = numpy.zeros(shape=(0,))

    for index in list_index:
        start = (index * n_patch)
        end = start + n_patch
        new_x = numpy.concatenate([new_x, x[start:end]])
        new_y = numpy.concatenate([new_y, y[start:end]])

    return new_x, new_y


def get_samples_train_and_test(index, x, y):
    return x[getattr(index, "index_test")], y[getattr(index, "index_test")], x[getattr(index, "index_train")], y[getattr(index, "index_train")]


def get_samples_and_labels(data):
    samples, n_features = data.shape
    return data[0:, 0:n_features - 1], data[:, n_features - 1]


def get_index(cfg, filename):
    x, y = get_samples_and_labels(numpy.loadtxt(filename))
    index = sklearn.model_selection.StratifiedShuffleSplit(n_splits=cfg["fold"], train_size=cfg["train_size"],
                                                           test_size=cfg["test_size"], random_state=1)
    return list([Index(i, index_train, index_test) for i, (index_train, index_test) in enumerate(index.split(x, y))])


@dataclasses.dataclass
class Index:
    fold: int
    index_train: list
    index_test: list

