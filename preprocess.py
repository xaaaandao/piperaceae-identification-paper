import dataclasses
import math
import numpy
import sklearn.decomposition
import sklearn.preprocessing


def get_all_values_pca(x):
    all_values_pca = 2 ** numpy.arange(7, math.floor(math.log2(x.shape[1])) + 1)
    return all_values_pca[all_values_pca < x.shape[0]]


def normalize_attributes(x):
    return sklearn.preprocessing.StandardScaler().fit_transform(x)


def has_more_attributes_value_pca(x):
    return True if x.shape[1] > 128 else False


def apply_pca(value, x):
    return sklearn.decomposition.PCA(n_components=value).fit_transform(x)


def get_data_with_pca(x):
    data_with_pca = list([Data(apply_pca(value_pca, x)) for value_pca in get_all_values_pca(x)])
    data_with_pca.append(Data(x))
    return data_with_pca


def preprocess(x):
    x = normalize_attributes(x)
    return get_data_with_pca(x) if has_more_attributes_value_pca(x) else list([Data(x)])


@dataclasses.dataclass
class Data:
    x: None
    y: None = dataclasses.field(init=False)
    n_labels: int = dataclasses.field(init=False)
    n_features: int = dataclasses.field(init=False)

    def __post_init__(self):
        self.n_features = self.x.shape[1]