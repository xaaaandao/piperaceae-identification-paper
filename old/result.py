import collections
import dataclasses
import numpy
import sklearn.metrics


def get_index_max_value(y):
    index = numpy.unravel_index(numpy.argmax(y, axis=None), y.shape)  # index return a tuple
    return int(index[1])


def max_rule(n_patch, y_pred):
    new_y_pred = numpy.empty(shape=(0,))
    for i, j in next_sequence(0, y_pred.shape[0], n_patch):
        new_y_pred = numpy.append(new_y_pred, get_index_max_value(y_pred[i:j]) + 1)
    return new_y_pred


def next_sequence(start, end, step):
    for i in range(start, end, step):
        yield i, i + step


def y_test_with_patch(n_patch, y_test):
    new_y_test = numpy.empty(shape=(0,))
    for i, j in next_sequence(0, y_test.shape[0], n_patch):
        new_y_test = numpy.append(new_y_test, y_test[i])
    return new_y_test


def y_pred_with_patch(n_patch, y_test):
    new_y_test = numpy.empty(shape=(0,))
    for i, j in next_sequence(0, y_test.shape[0], n_patch):
        new_y_test = numpy.append(new_y_test, collections.Counter(y_test[i:j].tolist()).most_common(1)[0][0])
    return new_y_test


def prod_all_prob(cfg, n_patch, y_pred):
    new_y_pred = numpy.empty(shape=(0,))
    new_y_pred_prob_prod = numpy.empty(shape=(0, cfg["n_labels"]))
    for i, j in next_sequence(0, y_pred.shape[0], n_patch):
        new_y_pred = numpy.append(new_y_pred, numpy.argmax(y_pred[i:j].prod(axis=0)) + 1)
        new_y_pred_prob_prod = numpy.vstack((new_y_pred_prob_prod, y_pred[i:j].prod(axis=0)))
    return new_y_pred_prob_prod, new_y_pred


def sum_all_prob(cfg, n_patch, y_pred):
    new_y_pred = numpy.empty(shape=(0,))
    new_y_pred_prob_sum = numpy.empty(shape=(0, cfg["n_labels"]))
    for i, j in next_sequence(0, y_pred.shape[0], n_patch):
        new_y_pred = numpy.append(new_y_pred, numpy.argmax(y_pred[i:j].sum(axis=0)) + 1)
        new_y_pred_prob_sum = numpy.vstack((new_y_pred_prob_sum, y_pred[i:j].sum(axis=0)))
    return new_y_pred_prob_sum, new_y_pred


def calculate_test(cfg, fold, y_pred, y_test, n_patch=1):
    if n_patch > 1:
        y_test = y_test_with_patch(n_patch, y_test)
    y_pred_max = max_rule(n_patch, y_pred)
    y_pred_prob_prod, y_pred_prod = prod_all_prob(cfg, n_patch, y_pred)
    y_pred_prob_sum, y_pred_sum = sum_all_prob(cfg, n_patch, y_pred)
    return Result(fold, "max", y_pred, y_pred_max, y_test), \
           Result(fold, "prod", y_pred_prob_prod, y_pred_prod, y_test), \
           Result(fold, "sum", y_pred_prob_sum, y_pred_sum, y_test)


def convert_prob_to_label(y_pred):
    y = numpy.empty(shape=(0,))
    for k, j in enumerate(y_pred):
        y = numpy.insert(y, k, [numpy.argmax(j) + 1])
    return y


def sum_all_results(list_result):
    list_result_per_sum = get_result_per_attribute_and_value("rule", list_result, "sum")
    result = getattr(list_result_per_sum[0], "y_pred_prob")
    for l in list_result_per_sum[1:]:
        result = result + getattr(l, "y_pred_prob")
    return result


def prod_all_results(list_result):
    list_result_per_prod = get_result_per_attribute_and_value("rule", list_result, "prod")
    result = getattr(list_result_per_prod[0], "y_pred_prob")
    for l in list_result_per_prod[1:]:
        result = result * getattr(l, "y_pred_prob")
    return result


def max_all_results(list_result):
    list_result_per_max = get_result_per_attribute_and_value("rule", list_result, "max")
    result = getattr(list_result_per_max[0], "y_pred")
    for y_pred in list_result_per_max[1:]:
        for row, current_y_pred in enumerate(result):
            result[row] = get_max_row_values(current_y_pred, row, getattr(y_pred, "y_pred"))
    return result


def get_max_row_values(current_y_pred, row, y_pred):
    return current_y_pred if numpy.all(current_y_pred > y_pred[row]) else y_pred[row]


def get_result_per_attribute_and_value(attribute, list_result_fold, value):
    return list(filter(lambda l: getattr(l, attribute) == value, list_result_fold))


@dataclasses.dataclass
class Result:
    accuracy: float = dataclasses.field(init=False)
    # classifier: c.Classifier
    confusion_matrix: None = dataclasses.field(init=False)
    fold: int
    rule: str
    y_pred_prob: None
    y_pred: None
    y_true: None

    def __post_init__(self):
        self.accuracy = sklearn.metrics.accuracy_score(y_pred=self.y_pred, y_true=self.y_true)
        self.confusion_matrix = sklearn.metrics.confusion_matrix(y_pred=self.y_pred, y_true=self.y_true)


def add_all_result(list_result_fold, list_result):
    for result in list_result:
        list_result_fold.append(result)
