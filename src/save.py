import logging
import os

import joblib
import numpy as np
import pandas as pd

class SaveExperiment:
    def __init__(self, experiment, output):
        self.experiment = experiment
        self.output = output
        os.makedirs(self.output, exist_ok=True)
        self.save()

    def save(self):
        self.save_best()
        self.save_info()
        self.save_folds()

    def save_best(self):
        output = os.path.join(self.output, "best")
        os.makedirs(output, exist_ok=True)

        self.save_best_fold(output)
        self.save_best_mean(output)

    def save_best_fold(self, output):
        data = self.experiment.best_fold.to_dict()
        filename = os.path.join(output, "best-fold.csv")
        df = pd.DataFrame(data)
        save_csv(df, filename)

    def save_best_mean(self, output):
        data = self.experiment.best_mean.to_dict()
        filename = os.path.join(output, "best-mean.csv")
        df = pd.DataFrame(data)
        save_csv(df, filename)

    def save_folds(self):
        for fold in self.experiment.folds:
            SaveFold(fold, self.output)

    def save_info(self):
        data = self.experiment.to_dict()
        filename = os.path.join(self.output, "experiment.csv")
        save_csv_transpose(data, filename, header=False, index=True)

class SaveFold:
    def __init__(self, fold, output):
        self.fold = fold
        self.output = os.path.join(output, "fold-%d" % fold.fold)
        os.makedirs(self.output, exist_ok=True)
        self.save()

    def save(self):
        self.save_best()
        self.save_data()
        self.save_idxs()
        self.save_predicts()
        self.save_results()

    def save_best(self):
        output = os.path.join(self.output, "best")
        os.makedirs(output, exist_ok=True)

        self.save_best_classifier(output)
        self.save_best_result(output)

    def save_best_classifier(self, output):
        self.save_best_classifier_cv_results(output)
        self.save_best_classifier_pkl(output)

    def save_best_classifier_pkl(self, output):
        filename = os.path.join(output, "best_classifier.pkl")
        logging.info("saved %s" % filename)

        try:
            with open(filename, "wb") as file:
                joblib.dump(self.fold.best_classifier, file, compress=3)
            file.close()
        except FileExistsError:
            logging.warning("problems in save model (%s)" % filename)

    def save_best_classifier_cv_results(self, output):
        filename = os.path.join(output, "best_classifier_cv_results.csv")
        data = self.fold.best_classifier.cv_results_
        df = pd.DataFrame(data)
        save_csv(df, filename, header=True, index=False)

    def save_best_result(self, output):
        data = self.fold.best_result.to_dict()
        filename = os.path.join(output, "best-result.csv")
        df = pd.DataFrame(data)
        save_csv(df, filename)

    def save_data(self):
        self.save_data_count_level()
        self.save_data_total()

    def save_data_count_level(self):
        data = self.fold.to_dict_data_count_level()
        filename = os.path.join(self.output, "fold-%d-count.csv" % self.fold.fold)
        df = pd.DataFrame(data)
        save_csv(df, filename)

    def save_data_total(self):
        data = self.fold.to_dict_data_total()
        filename = os.path.join(self.output, "fold-%d.csv" % self.fold.fold)
        df = pd.DataFrame(data)
        save_csv(df, filename)

    def save_idxs(self):
        output = os.path.join(self.output, "idx")
        os.makedirs(output, exist_ok=True)
        self.save_idx_train(output)
        self.save_idx_test(output)

    def save_idx_test(self, output):
        filename = os.path.join(output, "fold-%d-idx_test.npy" % self.fold.fold)
        np.save(filename, self.fold.idx.idx_test)
        logging.info("saved idx_test: %s" % filename)

    def save_idx_train(self, output):
        filename = os.path.join(output, "fold-%d-idx_train.npy" % self.fold.fold)
        np.save(filename, self.fold.idx.idx_train)
        logging.info("saved idx_train: %s" % filename)

    def save_predicts(self):
        for predict in self.fold.predicts:
            SavePredict(self.fold.fold, self.output, predict)

    def save_results(self):
        for result in self.fold.results:
            SaveResult(self.fold.fold, self.output, result)

class SavePredict:
    def __init__(self, fold, output, predict):
        self.fold = fold
        self.output = os.path.join(output, "results", predict.rule, "predicts")
        os.makedirs(self.output, exist_ok=True)
        self.predict = predict
        self.save()

    def save(self):
        self.save_y_pred()
        self.save_y_pred_proba()
        self.save_y_score()
        self.save_y_true()

    def save_y_pred(self):
        filename = os.path.join(self.output, "fold-%d-y_pred-%s.npy" % (self.fold, self.predict.rule))
        np.save(filename, self.predict.y_pred)

    def save_y_pred_proba(self):
        filename = os.path.join(self.output, "fold-%d-y_pred_proba-%s.npy" % (self.fold, self.predict.rule))
        np.save(filename, self.predict.y_pred_proba)

    def save_y_score(self):
        filename = os.path.join(self.output, "fold-%d-y_score-%s.npy" % (self.fold, self.predict.rule))
        np.save(filename, self.predict.y_score)

    def save_y_true(self):
        filename = os.path.join(self.output, "fold-%d-y_true-%s.npy" % (self.fold, self.predict.rule))
        np.save(filename, self.predict.y_true)

class SaveResult:
    def __init__(self, fold, output, result):
        self.fold = fold
        self.output = os.path.join(output, "results")
        os.makedirs(self.output, exist_ok=True)
        self.result = result
        self.save()

    def save(self):
        self.save_classification_report()
        self.save_confusion_matrix()
        self.save_metrics()
        self.save_topk()
        self.save_tp()

    def save_classification_report(self):
        output = os.path.join(self.output, self.result.rule, "classification_report")
        os.makedirs(output, exist_ok=True)

        filename = os.path.join(output, "fold-%d-classification_report-%s.csv" % (self.fold, self.result.rule))
        df = pd.DataFrame(self.result.classification_report)
        df = df.transpose()
        save_csv_transpose(df, filename, header=True, index=False)

    def save_metrics(self):
        data = self.result.to_dict_metrics()
        filename = os.path.join(self.output, self.result.rule, "fold-%d-%s.csv" % (self.fold, self.result.rule))
        df = pd.DataFrame(data, columns=data.keys())
        save_csv(df, filename, header=True, index=False)

    def save_topk(self):
        output = os.path.join(self.output, self.result.rule, "topk")
        os.makedirs(output, exist_ok=True)

        data = self.result.to_dict_top()
        filename = os.path.join(output, "fold-%d-topk-%s.csv" % (self.fold, self.result.rule))
        df = pd.DataFrame(data, columns=data.keys())
        save_csv(df, filename, header=True, index=False)

    def save_tp(self):
        output = os.path.join(self.output, self.result.rule, "true_positive")
        os.makedirs(output, exist_ok=True)

        data = self.result.to_dict_tp()
        filename = os.path.join(output, "fold-%d-true_positive-%s.csv" % (self.fold, self.result.rule))
        df = pd.DataFrame(data, columns=data.keys())
        save_csv(df, filename, header=True, index=False)

    def save_confusion_matrix(self):
        SaveConfusionMatrix(self.fold, self.output, self.result)

class SaveConfusionMatrix:
    def __init__(self, fold, output, result):
        self.confusion_matrix = result.confusion_matrix
        self.output = os.path.join(output, result.rule, "confusion-matrix")
        os.makedirs(self.output, exist_ok=True)
        self.save(fold, result)

    def save(self, fold, result):
        self.save_multilabel(fold, result.levels, self.output, result.rule)
        levels = ["%s+%s" % (l.name, l.label) for l in sorted(result.levels, key=lambda x: x.label)]
        self.save_normalized(fold, levels, self.output, result.rule)
        self.save_non_normalized(fold, levels, self.output, result.rule)

    def save_multilabel(self, fold, levels, output, rule):
        output_dir = os.path.join(output, "multilabel")
        os.makedirs(output_dir, exist_ok=True)

        for cm, level in zip(self.confusion_matrix.multilabel, sorted(levels, key=lambda x: x.label)):
            l = "%s+%s" % (level.name, level.label)
            filename = os.path.join(output_dir, "fold-%d-confusion_matrix_multilabel-%s-%s.csv" % (fold, l, rule))
            labels = ["True", "Negative"]
            df = pd.DataFrame(cm, index=labels, columns=labels)
            save_csv(df, filename, header=True, index=True)

    def save_normalized(self, fold, levels, output, rule):
        filename = os.path.join(output, "fold-%d-confusion_matrix_normalized-%s.csv" % (fold, rule))
        df = pd.DataFrame(self.confusion_matrix.normalized, index=levels, columns=levels)
        save_csv(df, filename, header=True, index=True)

    def save_non_normalized(self, fold, levels, output, rule):
        filename = os.path.join(output, "fold-%d-confusion_matrix_non_normalized-%s.csv" % (fold, rule))
        df = pd.DataFrame(self.confusion_matrix.non_normalized, index=levels, columns=levels)
        save_csv(df, filename, header=True, index=True)

def save_csv(df, filename, header=True, index=False):
    df.to_csv(filename, sep=";", quoting=2, index=index, header=header, encoding="utf-8")
    logging.info("saving %s" % filename)


def save_csv_transpose(data, filename, header, index):
    df = pd.DataFrame(data, columns=list(data.keys()))
    df = df.transpose()
    save_csv(df, filename, header=header, index=index)


def save(experiment, output):
    return SaveExperiment(experiment, output)
