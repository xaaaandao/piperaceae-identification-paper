from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import Config


def get_classifier(classifier, n_jobs, seed, verbose):
    """
    Retorna uma lista de objetos dos classificadores selecionados.
    :return: list, retorna uma lista com os classificadores (objetos) selecionados.
    """
    classifiers = [
        DecisionTreeClassifier(random_state=seed),
        KNeighborsClassifier(n_jobs=n_jobs),
        MLPClassifier(random_state=seed),
        RandomForestClassifier(random_state=seed, n_jobs=n_jobs, verbose=verbose, max_depth=10),
        SVC(random_state=seed, verbose=verbose, cache_size=2000, C=0.001)
    ]
    for c in classifiers:
        if classifier == c.__class__.__name__:
            return c
    raise ValueError("classifier %s not found" % classifier)