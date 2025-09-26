import collections
import functools
import itertools
import logging
import os

import joblib
import numpy as np
import pandas as pd



class SaveExperiment:
    def __init__(self, experiment, folds, output):
        self.experiment = experiment
        self.folds = folds
        self.rules = ["sum", "max", "mult"]
        self.output = output
        os.makedirs(self.output, exist_ok=True)

        self.bests()
        self.cv_fold()
        self.infos()
        self.means()

    def bests(self):
        output_dir = os.path.join(self.output, "best")
        os.makedirs(output_dir, exist_ok=True)

        self.best_mean(output_dir)
        self.best_fold(output_dir)

    def best_mean(self, output):
        filename = os.path.join(output, "best_means.csv")
        data = {
            "best_f1": [self.experiment.best_mean.f1],
            "best_f1_std": [self.experiment.best_mean.f1_std],
            "best_f1_rule": [self.experiment.best_mean.f1_rule],
            "best_accuracy": [self.experiment.best_mean.accuracy],
            "best_accuracy_std": [self.experiment.best_mean.accuracy_std],
            "best_accuracy_rule": [self.experiment.best_mean.accuracy_rule]
        }
        save_csv_transpose(data, filename, header=False, index=True)

    def best_fold(self, output):
        filename = os.path.join(output, "best_fold.csv")
        # TODO ta estranho
        data = {
            "best_fold_f1": [self.experiment.best_fold.f1],
            # "best_fold_f1_rule": [self.experiment.best_fold.rule],
            # "best_fold_f1_fold": [self.experiment.best_fold],
            "best_fold_accuracy": [self.experiment.best_fold.accuracy],
            # "best_fold_accuracy_rule": [self.experiment.best_fold.rule],
            # "best_fold_accuracy_fold": [self.experiment.fold],
        }
        save_csv_transpose(data, filename, header=False, index=True)

    def cv_fold(self):
        for f in self.folds:
            output_dir = os.path.join(self.output, "fold-%d" % f.fold)
            os.makedirs(output_dir, exist_ok=True)
            f.save(output_dir)

    def infos(self):
        filename = os.path.join(self.output, "experiment.csv")
        data = {
            "backend": [self.experiment.backend],
            "classifier": [self.experiment.classifier.__class__.__name__],
            "cv_metric": [self.experiment.cv_metric],
            "folds": [self.experiment.folds],
            "input": [self.experiment.dataset.input_dir],
            "model": [self.experiment.dataset.model],
            "metrics": [str(self.experiment.metrics)],
            "n_features": [self.experiment.dataset.n_features],
            "n_jobs": [self.experiment.n_jobs],
            "n_label": [self.experiment.dataset.n_labels],
            "n_samples": [self.experiment.dataset.n_samples],
            "seed": [self.experiment.seed],
            "verbose": [self.experiment.verbose]
        }
        save_csv_transpose(data, filename, header=False, index=True)

    def means(self):
        output_dir = os.path.join(self.output, "mean")
        os.makedirs(output_dir, exist_ok=True)

        self.mean_f1_accuracy(output_dir)
        self.mean_topk(output_dir)
        self.mean_tps(output_dir)

    def mean_f1_accuracy(self, output):
        filename = os.path.join(output, "means.csv")
        df = pd.DataFrame([mean.to_dict() for mean in self.experiment.means])
        save_csv(df, filename)

    def mean_topk(self, output):
        output_dir = os.path.join(output, "topk")
        os.makedirs(output_dir, exist_ok=True)
        mean_test = [f.total_test_no_patch for f in self.folds]

        for rule in self.rules:
            filename = os.path.join(output_dir , "means+topk+%s.csv" % rule)
            topks = [m.topks for m in self.experiment.means if m.rule == rule]
            topks = list(itertools.chain(*topks))
            data = {
                "k": [t.k for t in topks],
                "top_k_accuracy_score": [t.mean for t in topks],
                "top_k_accuracy_score_std": [t.std for t in topks],
                "mean_test": np.mean(mean_test),
            }
            df = pd.DataFrame(data)
            save_csv(df, filename)

    def mean_tps(self, output):
        output_dir = os.path.join(output, "true_positive")
        os.makedirs(output_dir, exist_ok=True)
        tests = [f.count_test for f in self.folds]
        count_test = dict(functools.reduce(lambda x, y: collections.Counter(x) + collections.Counter(y), tests))

        for rule in self.rules:
            filename = os.path.join(output_dir , "means+true_positive+%s.csv" % rule)
            tps = [m.true_positives for m in self.experiment.means if m.rule == rule]
            tps = list(itertools.chain(*tps))
            data = {
                "label": [t.label for t in tps],
                "specific_epithet": [t.specific_epithet for t in tps],
                "true_positive": [t.mean for t in tps],
                "true_positive_std": [t.std for t in tps],
                "mean_test": [(count_test[t.label] / self.experiment.dataset.patch) / len(tests) for t in tps],
            }
            df = pd.DataFrame(data)
            df.to_csv(filename, sep=";", quoting=2, index=False, encoding="utf-8")
            logging.info("saving %s" % filename)

class SaveFold:
    def __init__(self, fold, output):
        self.fold = fold
        self.output = output

        self.best()
        self.counts()
        self.idxs()
        self.info()
        self.results()

    def best(self):
        output_dir = os.path.join(self.output, "best")
        os.makedirs(output_dir, exist_ok=True)

        self.best_classifier(output_dir)
        self.best_results(output_dir)

    def best_classifier(self, output):
        self.best_classifier_cv_results(output)
        self.best_classifier_pkl(output)

    def best_classifier_pkl(self, output):
        filename = os.path.join(output, "fold-%d-best_classifier.pkl" % self.fold.fold)
        logging.info("saving %s" % filename)

        try:
            with open(filename, "wb") as file:
                joblib.dump(self.best_classifier, file, compress=3)
            file.close()
        except FileExistsError:
            logging.warning("problems in save model (%s)" % filename)

    def best_classifier_cv_results(self, output):
        filename = os.path.join(output, "fold-%d-best_classifier.csv" % self.fold.fold)
        df = pd.DataFrame(self.fold.best_classifier.cv_results_)
        save_csv(df, filename, header=True, index=False)

    def best_results(self, output):
        filename = os.path.join(output, "fold-%d-best_results.csv" % self.fold.fold)
        data = {
            "best_f1": [self.fold.best_result.f1],
            "best_f1_rule": [self.fold.best_result.f1_rule],
            "best_accuracy": [self.fold.best_result.accuracy],
            "best_accuracy_rule": [self.fold.best_result.accuracy_rule]
        }
        save_csv_transpose(data, filename, header=False, index=True)

    def counts(self):
        filename = os.path.join(self.output, "fold-%d-count.csv" % self.fold.fold)
        data = []
        for l in self.fold.dataset.levels:
            data.append({"label": l.label,
                         "specific_epithet": l.specific_epithet,
                         "count_train": self.fold.count_train[l.label] / self.fold.dataset.patch,
                         "count_test": self.fold.count_test[l.label] / self.fold.dataset.patch,
            })
        df = pd.DataFrame(data)
        save_csv(df, filename, header=True, index=False)

    def idxs(self):
        output_dir = os.path.join(self.output, "idx")
        os.makedirs(output_dir, exist_ok=True)

        self.idx_train(output_dir)
        self.idx_test(output_dir)

    def idx_train(self, output: str):
        filename = os.path.join(output, "fold-%d-idx_train.npy" % self.fold.fold)
        np.save(filename, self.fold.idx_train)
        logging.info("saving %s" % filename)

    def idx_test(self, output_dir: str):
        filename = os.path.join(output_dir, "fold-%d-idx_test.npy" % self.fold.fold)
        np.save(filename, self.fold.idx_test)
        logging.info("saving %s" % filename)

    def info(self):
        filename = os.path.join(self.output, "fold-%d.csv" % self.fold.fold)
        data = {
            "total_test": [self.fold.total_test],
            "total_train": [self.fold.total_train],
            "total_test_no_patch": [self.fold.total_test_no_patch],
            "total_train_no_patch": [self.fold.total_train_no_patch],
        }
        save_csv_transpose(data, filename, header=False, index=True)

    def results(self):
        output_dir = os.path.join(self.output, "results")
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.join(output_dir, "fold-%d-results.csv" % self.fold.fold)
        df = pd.DataFrame([result.to_dict() for result in self.fold.results])
        save_csv(df, filename, header=True, index=False)

        for result in self.fold.results:
            output_dir = os.path.join(self.output, "results", result.rule)
            os.makedirs(output_dir, exist_ok=True)

            # TODO SaveResult
            result.save_predictions(self.fold.fold, output_dir)
            result.save_confusion_matrix(self.fold.fold, output_dir)
            result.save_classification_report(self.fold.fold, output_dir)
            result.save_topk(self.fold.fold, output_dir, self.fold.total_test_no_patch)
            result.save_tp(self.fold.count_test, self.fold.fold, output_dir, self.fold.dataset.patch, self.fold.total_test_no_patch)

def save_csv(df: pd.DataFrame, filename: str, header=True, index=False):
    df.to_csv(filename, sep=";", quoting=2, index=index, header=header, encoding="utf-8")
    logging.info("saving %s" % filename)

def save_csv_transpose(data, filename, header, index):
    df = pd.DataFrame(data, columns=list(data.keys()))
    df = df.transpose()
    save_csv(df, filename, header=header, index=index)
