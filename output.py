import csv
import matplotlib
import numpy
import os
import pandas
import pathlib
import sklearn
import time


def save_confusion_matrix(classifier_name, dataset, path, result):
    # for result in list_result:
    filename = f"confusion_matrix-{result['rule']}.png"
    labels = ["$\it{Manekia}$", "$\it{Ottonia}$", "$\it{Peperomia}$", "$\it{Piper}$", "$\it{Pothomorphe}$"]
    confusion_matrix = sklearn.metrics.ConfusionMatrixDisplay(result["confusion_matrix"])
    confusion_matrix.plot(cmap="Reds")
    title = f"Confusion Matrix\ndataset: {dataset}, classifier: {classifier_name}\naccuracy: {round(result['accuracy'] * 100, 4)}, rule: {result['rule']}"
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


def save(best_params, cfg, classifier_name, dataset, list_result_fold, list_time, path):
    save_fold(cfg, classifier_name, dataset, list_result_fold, list_time, path)
    save_mean(best_params, list_result_fold, list_time, path)


def save_mean(best_params, list_result_fold, list_time, path):
    best_fold = max(list_result_fold, key=lambda x: x["accuracy"])
    mean_time = numpy.mean([t["final_time"] for t in list_time])
    mean_time_sec = time.strftime("%H:%M:%S", time.gmtime(mean_time))
    std_time = numpy.std([t["final_time"] for t in list_time])
    print(f"best acc (%): {round(best_fold['accuracy'] * 100, 4)}")
    print(f"best fold: {best_fold['fold']}, best rule: {best_fold['rule']}")
    dataframe_mean = pandas.DataFrame(
        [mean_time, mean_time_sec, std_time, best_fold["fold"], best_fold["rule"], best_fold["accuracy"],  round(best_fold["accuracy"] * 100, 4), str(best_params)],
        ["mean_time", "mean_time_sec", "std_time", "best_fold", "best_rule", "best_fold_accuracy", "best_fold_accuracy_per", "best_params"])
    dataframe_mean.to_csv(os.path.join(path, "mean.csv"), decimal=",", sep=";", na_rep=" ", header=False,
                          quoting=csv.QUOTE_ALL)


def save_fold(cfg, classifier_name, dataset, list_result_fold, list_time, path):
    columns = ["rule", "accuracy", "accuracy_per"]
    for f in range(0, cfg["fold"]):
        list_fold = list(filter(lambda x: x["fold"] == f, list_result_fold))
        t = list(filter(lambda x: x["fold"] == f, list_time))

        list_rule = list()
        list_accuracy = list()
        list_accuracy_per = list()
        path_fold = os.path.join(path, str(f))

        pathlib.Path(path_fold).mkdir(parents=True, exist_ok=True)
        for rule in list(["max", "prod", "sum"]):
            result = list(filter(lambda x: x["rule"] == rule, list_fold))

            if len(result) > 0:
                r = result[0]
            else:
                r = list_fold[0]

            list_rule.append(rule)
            list_accuracy.append(r["accuracy"])

            list_accuracy_per.append(round(r["accuracy"] * 100, 4))
            save_confusion_matrix(classifier_name, dataset, path_fold, r)

        best_rule = max(list_fold, key=lambda x: x["accuracy"])

        dataframe_fold = pandas.DataFrame([list_rule, list_accuracy, list_accuracy_per], columns)
        dataframe_fold.to_csv(os.path.join(path_fold, "out.csv"), decimal=",", sep=";", na_rep=" ", header=False,
                              quoting=csv.QUOTE_ALL)

        time_sec = time.strftime("%H:%M:%S", time.gmtime(t[0]["final_time"]))
        dataframe_time = pandas.DataFrame([t[0]["final_time"], time_sec], ["time", "time_sec"])
        dataframe_best_rule = pandas.DataFrame([best_rule["rule"], best_rule["accuracy"]],
                                               ["best_rule", "best_accuracy"])
        dataframe_info = pandas.concat([dataframe_time, dataframe_best_rule])
        dataframe_info.to_csv(os.path.join(path_fold, "fold_info.csv"), decimal=",", sep=";", na_rep=" ", header=False,
                              quoting=csv.QUOTE_ALL)
