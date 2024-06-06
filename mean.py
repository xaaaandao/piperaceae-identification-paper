import itertools
import logging

import numpy as np
import pandas as pd

from evaluate import TopK


class Mean:
    mean_f1: float
    std_f1: float
    mean_accuracy: float
    std_accuracy: float
    mean_topk: list
    std_topk: list
    mean_time: float
    std_time: float
    mean_true_positive: list
    std_true_positive: list
    rule: str

    def __init__(self, folds: list, levels: list, rule: str):
        results = [fold.result for fold in folds]
        predicts = [result.predicts for result in results]
        evaluations = [p.eval for p in list(itertools.chain(*predicts)) if p.rule.__eq__(rule)]
        self.mean_count_test = np.mean([result.total_test_no_patch for result in results])
        self.rule = rule
        self.set_time(results)
        self.set_true_positive(evaluations)
        self.set_f1(evaluations)
        self.set_accuracy(evaluations)
        self.set_topk(evaluations, levels)

    def set_time(self, results):
        self.mean_time = np.mean([result.time for result in results])
        self.std_time = np.std([result.time for result in results])

    def set_f1(self, evaluations: list):
        self.mean_f1 = np.mean([evaluation.f1 for evaluation in evaluations])
        self.std_f1 = np.std([evaluation.f1 for evaluation in evaluations])
        logging.info('Mean F1 score %.2f and std F1 score %.2f', self.mean_f1, self.std_f1)

    def set_accuracy(self, evaluations: list):
        self.mean_accuracy = np.mean([evaluation.accuracy for evaluation in evaluations])
        self.std_accuracy = np.std([evaluation.accuracy for evaluation in evaluations])
        logging.info('Mean accuracy score %.2f and std accuracy score %.2f', self.mean_accuracy, self.std_accuracy)

    def set_topk(self, evaluations: list, levels: list):
        self.mean_topk, self.std_topk = [], []
        for evaluation in evaluations:
            for k in range(3, len(levels)):
                values = list(filter(lambda x: x.k.__eq__(k), evaluation.topk))
                self.mean_topk.append(TopK(k, topk=np.mean([v.top_k_accuracy_score for v in values])))
                self.std_topk.append(TopK(k, topk=np.std([v.top_k_accuracy_score for v in values])))

    def save_topk(self) -> pd.DataFrame:
        data = {
            'k': [topk.k for topk in sorted(self.mean_topk, key=lambda x: x.k)],
            'mean': [topk.top_k_accuracy_score for topk in sorted(self.mean_topk, key=lambda x:x.k)],
            'mean_count_test': np.repeat(self.mean_count_test, len(self.mean_topk)),
            'std': [topk.top_k_accuracy_score for topk in sorted(self.std_topk, key=lambda x:x.k)],
            'rule': [self.rule] * len(self.mean_topk),
            'mean+100': [topk.top_k_accuracy_score/self.mean_count_test for topk in sorted(self.mean_topk, key=lambda x:x.k)],
        }
        return pd.DataFrame(data, columns=data.keys())

    def save(self):
        data = {
            'mean': [self.mean_f1, self.mean_accuracy, self.mean_time],
            'std': [self.std_f1, self.std_accuracy, self.std_time],
            'metric': ['f1', 'accuracy', 'time'],
            'rule': [self.rule, self.rule, '']
        }
        return pd.DataFrame(data, columns=data.keys())

    def set_true_positive(self, evaluations):
        self.true_positive = [np.diag(evaluation.confusion_matrix) for evaluation in evaluations]
        self.mean_true_positive = np.mean(self.true_positive, axis=0)
        self.std_true_positive = np.std(self.true_positive, axis=0)

    def save_true_positive(self, levels):
        data = {
            'labels': ['%s+%s' % (l.specific_epithet, l.label) for l in sorted(levels, key=lambda x: x.label)],
            'mean': self.mean_true_positive,
            'std': self.std_true_positive,
            'rule': [self.rule] * len(self.mean_true_positive),
        }
        return pd.DataFrame(data, columns=data.keys())