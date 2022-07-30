import dataclasses
import sklearn.ensemble
import sklearn.exceptions
import sklearn.model_selection
import sklearn.neighbors
import sklearn.neural_network
import sklearn.svm
import sklearn.tree
import warnings

warnings.simplefilter("ignore", category=sklearn.exceptions.ConvergenceWarning)


def get_hp_dt():
    return {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [10, 100, 1000]
    }


def get_hp_svm():
    return {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    }


def get_hp_mlp():
    return {
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['adam', 'sgd'],
        'learning_rate_init': [0.01, 0.001, 0.0001],
        'momentum': [0.9, 0.4, 0.1]
    }


def get_hp_knn():
    return {
        'n_neighbors': [2, 4, 6, 8, 10],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }


def get_hp_rf():
    return {
        'n_estimators': [200, 400, 600, 800, 1000],
        'max_features': ['sqrt', 'log2'],
        'criterion': ['gini', 'entropy'],
        'max_depth': [10, 100, 1000]
    }


def get_list_classifiers():
    return list([
        Classifier("dt", sklearn.tree.DecisionTreeClassifier(random_state=1), get_hp_dt(), None, None),
        Classifier("knn", sklearn.neighbors.KNeighborsClassifier(n_jobs=-1), get_hp_knn(), None, None),
        Classifier("mlp", sklearn.neural_network.MLPClassifier(random_state=1), get_hp_mlp(), None, None),
        Classifier("rf", sklearn.ensemble.RandomForestClassifier(random_state=1, n_jobs=-1), get_hp_rf(), None, None),
        Classifier("svm", sklearn.svm.SVC(random_state=1, probability=True), get_hp_svm(), None, None),
    ])


def train_and_test(model, x_test, x_train, y_train):
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)


def get_best_classifier(cfg, classifier, x, y):
    model = sklearn.model_selection.GridSearchCV(getattr(classifier, "classifier"), getattr(classifier, "params"),
                                                 scoring='accuracy', cv=cfg["fold"])
    model.fit(x, y)
    setattr(classifier, "best_classifier", model.best_estimator_)
    setattr(classifier, "best_params", model.best_params_)


@dataclasses.dataclass
class Classifier:
    name: str
    classifier: None
    params: dict
    best_classifier: None
    best_params: dict
