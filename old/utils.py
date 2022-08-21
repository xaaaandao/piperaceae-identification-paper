import numpy
import os
import sklearn.model_selection


def get_cfg():
    return {
        "fold": 5,
        "n_labels": 5,
        "path_base": "../dataset",
        "path_out": "out",
        "test_size": 0.2,
        "train_size": 0.8,
    }


def get_index():
    cfg = get_cfg()
    surf = numpy.loadtxt(os.path.join(cfg["path_base"], "surf64.txt"))
    n_samples, n_features = surf.shape
    x, y = surf[0:, 0:n_features - 1], surf[:, n_features - 1]
    index = sklearn.model_selection.StratifiedShuffleSplit(n_splits=cfg["fold"], train_size=cfg["train_size"],
                                                           test_size=cfg["test_size"], random_state=1)
    return list(index.split(x, y))
