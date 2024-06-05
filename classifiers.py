from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import Config


def get_classifiers() -> list:
    return ['DecisionTreeClassifier', 'RandomForestClassifier', 'KNeighborsClassifier', 'MLPClassifier', 'SVC']


def select_classifiers(config: Config, selected: list) -> list:
    classifiers = [
        DecisionTreeClassifier(random_state=config.seed),
        KNeighborsClassifier(n_jobs=config.n_jobs),
        MLPClassifier(random_state=config.seed),
        RandomForestClassifier(random_state=config.seed, n_jobs=config.n_jobs, verbose=config.verbose, max_depth=10),
        SVC(random_state=config.seed, verbose=config.verbose, cache_size=2000, C=0.01)
    ]
    return [c for cs in selected for c in classifiers if cs == c.__class__.__name__]
