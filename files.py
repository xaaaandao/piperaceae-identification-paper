import time

import matplotlib.pyplot
import numpy
import operator
import os
import pathlib
import sklearn.metrics

from result import Result, get_result_per_attribute_and_value


def create_outfile_each_fold(elapsed_time, list_result, path):
    # result/cnn/classifier/pca/fold
    try:
        for result in list_result:
            if getattr(result, "rule") == "sum":
                filename = f"out+{getattr(result, 'rule')}.txt"
                classifier = getattr(result, "classifier")
                with open(os.path.join(path, filename), "w") as file:
                    file.write(f"fold: {getattr(result, 'fold')} \n")
                    file.write(f"rule: {getattr(result, 'rule')}, elapsed time: {elapsed_time}\n")
                    file.write(f"classifier: {getattr(classifier, 'name')}, best_params: {getattr(classifier, 'best_params')}\n")
                    file.write(f"accuracy: {getattr(result, 'accuracy')}\n")
                    file.write(f"accuracy (%): {round(getattr(result, 'accuracy') * 100, 4)}\n")
                    print(f"type: {getattr(result, 'rule')}, accuracy (%): {round(getattr(result, 'accuracy') * 100, 4)}")
                    file.write(f"confusion matrix:\n{getattr(result, 'confusion_matrix')}\n")
                file.close()
                # print(f"file created {os.path.join(path, filename)}")
    except Exception as e:
        print(f"exception in {e}")
        raise


def create_outfile_mean_fold(list_elapsed_time, list_result_fold, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    list_result_between_rule = list([])
    try:
        with open(os.path.join(path, "out.txt"), "w") as file:
            for rule in ("max", "prod", "sum"):
                if rule == "sum":
                    list_result_per_rule = get_result_per_attribute_and_value("rule", list_result_fold, rule)
                    mean_accuracy = numpy.mean(get_all_by_attribute("accuracy", list_result_per_rule))
                    mean_elapsed_time = numpy.mean(list_elapsed_time)
                    std_deviation = numpy.std(get_all_by_attribute("accuracy", list_result_per_rule))
                    best_fold = max(list_result_fold, key=operator.attrgetter("accuracy"))
                    file.write(f"rule: {rule}, mean elapsed time: {time.strftime('%H:%M:%S', time.gmtime(mean_elapsed_time))}\n")
                    file.write(f"mean accuracy (%): {round(mean_accuracy * 100, 4)}, std deviation: {round(std_deviation, 4)}\n")
                    file.write(f"best fold: {getattr(best_fold, 'fold')}, best accuracy: {getattr(best_fold, 'accuracy')}\n")
                    print(f"mean accuracy (%): {round(mean_accuracy * 100, 4)}, std deviation: {round(std_deviation, 4)}, rule: {rule}, mean elapsed time: {time.strftime('%H:%M:%S', time.gmtime(mean_elapsed_time))} ({mean_elapsed_time})")
                    file.write("---------------------------------\n")
                    result = Result(None, None, rule, numpy.zeros(shape=(1,)), numpy.zeros(shape=(1,)), numpy.zeros(shape=(1,)))
                    setattr(result, "accuracy", mean_accuracy)
                    list_result_between_rule.append(result)
            best_rule = max(list_result_between_rule, key=operator.attrgetter("accuracy"))
            file.write(f"best_accuracy: {round(getattr(best_rule, 'accuracy') * 100, 4)} rule:{getattr(best_rule, 'rule')}\n")
            print(f"best_accuracy: {round(getattr(best_rule, 'accuracy') * 100, 4)} rule:{getattr(best_rule, 'rule')}\n")
            file.close()
    except Exception as e:
        print(f"exception in {e}")
        raise


def get_all_by_attribute(attribute, list_result):
    return list([getattr(l, attribute) for l in list_result])


def get_title(accuracy, classifier, dataset, rule):
    return f"Confusion Matrix\ndataset: {dataset}, classifier:{classifier}\naccuracy: {round(accuracy * 100, 4)}, rule: {rule}"


def plot_confusion_matrix(classifier, dataset, list_result, path):
    for result in list_result:
        if getattr(result, "rule") == "sum":
            filename = f"confusion_matrix+{getattr(result, 'rule')}.png"
            labels = ["$\it{Manekia}$", "$\it{Ottonia}$", "$\it{Peperomia}$", "$\it{Piper}$", "$\it{Pothomorphe}$"]
            confusion_matrix = sklearn.metrics.ConfusionMatrixDisplay(getattr(result, "confusion_matrix"))
            confusion_matrix.plot(cmap="Reds")
            title = get_title(getattr(result, "accuracy"), classifier, dataset, getattr(result, "rule"))
            matplotlib.pyplot.ioff()
            matplotlib.pyplot.title(title, pad=20)
            matplotlib.pyplot.xticks(numpy.arange(5), labels, rotation=(45))
            matplotlib.pyplot.yticks(numpy.arange(5), labels)
            matplotlib.pyplot.ylabel("y_test", fontsize=12)
            matplotlib.pyplot.xlabel("y_pred", fontsize=12)
            matplotlib.pyplot.gcf().subplots_adjust(bottom=0.15, left=0.25)
            matplotlib.pyplot.savefig(os.path.join(path, filename))
            matplotlib.pyplot.cla()
            matplotlib.pyplot.clf()
            matplotlib.pyplot.close()


def get_path_each_fold(cfg, classifier, dataset, fold, pca):
    return os.path.join(cfg["path_out"], getattr(dataset, "name"), getattr(classifier, "name"), fold, pca)
