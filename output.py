import csv
import markdownTable
import matplotlib
import numpy
import operator
import os
import pandas
import re
import sklearn
import time

from result import Result


def save_mean_std(best_params, cfg, list_result_fold, list_time, n_features, n_samples, path):
    columns_cfg = ["fold", "n_labels", "path_base", "path_out", "test_size", "train_size"]
    data_cfg = [cfg["fold"], cfg["n_labels"], cfg["path_base"], cfg["path_out"], cfg["test_size"], cfg["train_size"]]
    columns_best_params = ["best_params"]
    data_best_params = [cfg["best_params"]]
    columns_dataset = ["n_samples", "n_features"]
    data_dataset = [n_samples, n_features]
    cfg_used = {
        "fold": str(cfg["fold"]),
        "n_labels": str(cfg["n_labels"]),
        "path_base": str(cfg["path_base"]),
        "path_out": str(cfg["path_out"]),
        "test_size": str(cfg["test_size"]),
        "train_size": str(cfg["train_size"])
    }
    b_params = {
        "best_params": str(best_params)
    }
    info_dataset = {
        "n_samples": str(n_samples),
        "n_features": str(n_features)
    }
    try:
        with open(os.path.join(path, "mean.md"), "w") as file:
            file.write(re.sub(r"```$", "\n```\n\n", markdownTable.markdownTable(list([cfg_used])).getMarkdown()))
            file.write(re.sub(r"```$", "\n```\n\n", markdownTable.markdownTable(list([info_dataset])).getMarkdown()))
            file.write(re.sub(r"```$", "\n```\n\n", markdownTable.markdownTable(list([b_params])).getMarkdown()))

            list_result_per_rule = list(r for r in list_result_fold if not getattr(r, "rule"))
            if len(list_result_per_rule) == 5:
                mean_time = numpy.mean(list_time)
                mean_accuracy = numpy.mean(list(getattr(l, "accuracy") for l in list_result_per_rule))
                std_deviation = numpy.std(list(getattr(l, "accuracy") for l in list_result_per_rule))
                columns_mean = ["mean_time", "mean_accuracy", "mean_accuracy_per", "std_deviation"]
                data_mean = [str(time.strftime("%H:%M:%S", time.gmtime(mean_time))), mean_accuracy,
                             round(mean_accuracy * 100, 4), std_deviation]
                # mean = {
                #     "mean_time": ,
                #     "mean_accuracy": str(mean_accuracy),
                #     "mean_accuracy_per": str(round(mean_accuracy * 100, 4)),
                #     "std_deviation": str(std_deviation),
                # }
                # file.write(re.sub(r"```$", "\n```\n\n", markdownTable.markdownTable(list([mean])).getMarkdown()))
                best_accuracy = max(list_result_per_rule, key=operator.attrgetter("accuracy"))
                columns_best = ["best_fold", "best_accuracy", "best_accuracy_per"]
                data_best = [getattr(best_accuracy, "fold"), getattr(best_accuracy, "accuracy"),
                             round(getattr(best_accuracy, "accuracy") * 100, 4)]
                # best = {
                #     "best_fold": str(getattr(best_accuracy, "fold")),
                #     "best_accuracy": str(getattr(best_accuracy, "accuracy")),
                #     "best_accuracy_per": str(round(getattr(best_accuracy, "accuracy") * 100, 4)),
                # }
                # print(f"best_accuracy: {best['best_accuracy_per']}\n")
                # file.write(re.sub(r"```$", "\n```\n\n", markdownTable.markdownTable(list([best])).getMarkdown()))
            else:
                list_result_between_rule = list()
                for rule in ("max", "prod", "sum"):
                    file.write(f"### {rule}\n")
                    list_result_per_rule = list(filter(lambda l: getattr(l, "rule") == rule, list_result_fold))
                    mean_time = numpy.mean(list_time)
                    mean_accuracy = numpy.mean(list(getattr(l, "accuracy") for l in list_result_per_rule))
                    std_deviation = numpy.std(list(getattr(l, "accuracy") for l in list_result_per_rule))
                    best_fold = max(list_result_per_rule, key=operator.attrgetter("accuracy"))

                    mean = {
                        "mean_time": str(time.strftime("%H:%M:%S", time.gmtime(mean_time))),
                        "mean_accuracy": str(mean_accuracy),
                        "mean_accuracy_per": str(round(mean_accuracy * 100, 4)),
                        "std_deviation": str(std_deviation),
                    }
                    columns_mean = ["mean_time", "mean_accuracy", "mean_accuracy_per", "std_deviation"]
                    data_mean = [str(time.strftime("%H:%M:%S", time.gmtime(mean_time))), mean_accuracy,
                                 round(mean_accuracy * 100, 4), std_deviation]
                    best = {
                        "best_fold": str(getattr(best_fold, "fold")),
                        "best_accuracy": str(getattr(best_fold, "accuracy")),
                        "best_accuracy_per": str(round(getattr(best_fold, "accuracy") * 100, 4)),
                    }
                    columns_b = ["best_fold", "best_accuracy", "best_accuracy_per"]
                    data_b = [getattr(best_fold, "fold"), getattr(best_fold, "accuracy"),
                              round(getattr(best_fold, "accuracy") * 100, 4)]

                    file.write(re.sub(r"```$", "\n```\n\n", markdownTable.markdownTable(list([mean])).getMarkdown()))
                    file.write(re.sub(r"```$", "\n```\n\n", markdownTable.markdownTable(list([best])).getMarkdown()))

                    result = Result(None, rule, numpy.zeros(shape=(1,)), numpy.zeros(shape=(1,)),
                                    numpy.zeros(shape=(1,)))
                    setattr(result, "accuracy", mean_accuracy)
                    list_result_between_rule.append(result)
                best_rule = max(list_result_between_rule, key=operator.attrgetter("accuracy"))
                b = {
                    "best_accuracy": str(round(getattr(best_rule, "accuracy") * 100, 4)),
                    "best_rule": str(getattr(best_rule, "rule"))
                }
                print(f"best_accuracy: {b['best_accuracy']}, best_rule: {b['best_rule']}\n")
                file.write("### final\n")
                file.write(re.sub(r"```$", "\n```\n\n", markdownTable.markdownTable(list([b])).getMarkdown()))
            file.close()
    except Exception as e:
        print(f"exception in {e}")
        raise


def save_confusion_matrix(classifier_name, dataset, list_result, path):
    for result in list_result:
        filename = f"confusion_matrix-{getattr(result, 'rule')}.png"
        labels = ["$\it{Manekia}$", "$\it{Ottonia}$", "$\it{Peperomia}$", "$\it{Piper}$", "$\it{Pothomorphe}$"]
        confusion_matrix = sklearn.metrics.ConfusionMatrixDisplay(getattr(result, "confusion_matrix"))
        confusion_matrix.plot(cmap="Reds")
        title = f"Confusion Matrix\ndataset: {dataset}, classifier: {classifier_name}\naccuracy: {round(getattr(result, 'accuracy') * 100, 4)}, rule: {getattr(result, 'rule')}"
        matplotlib.pyplot.ioff()
        matplotlib.pyplot.title(title, pad=20)
        matplotlib.pyplot.xticks(numpy.arange(5), labels, rotation=(45))
        matplotlib.pyplot.yticks(numpy.arange(5), labels)
        matplotlib.pyplot.ylabel("y_test", fontsize=12)
        matplotlib.pyplot.xlabel("y_pred", fontsize=12)
        matplotlib.pyplot.gcf().subplots_adjust(bottom=0.15, left=0.25)
        matplotlib.pyplot.rcParams["figure.facecolor"] = "white"
        matplotlib.pyplot.rcParams["figure.figsize"] = (10, 10)
        matplotlib.pyplot.savefig(os.path.join(path, filename))
        matplotlib.pyplot.cla()
        matplotlib.pyplot.clf()
        matplotlib.pyplot.close()


def save_fold(classifier_name, dataset, final_time, list_result, path):
    columns = ["fold", "rule", "accuracy", "accuracy_per", "time", "time_seg"]
    for result in list_result:
        data = [getattr(result, "fold"), getattr(result, "rule"), getattr(result, "accuracy"),
                round(getattr(result, "accuracy") * 100, 4), str(time.strftime("%H:%M:%S", time.gmtime(final_time))),
                final_time]
        dataframe_cfg = pandas.DataFrame(data, columns)
        dataframe_cfg.to_csv(os.path.join(path, "out.csv"), decimal=",", sep=";", na_rep=" ", header=False,
                             quoting=csv.QUOTE_ALL)
        save_confusion_matrix(classifier_name, dataset, list_result, path)
