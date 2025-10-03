import os

import numpy as np


class Predict:
    def __init__(self, patch, rule, y_pred_proba, y_test):
        self.num_groups = y_pred_proba.shape[0] // patch
        self.patch = patch
        self.rule = rule
        self.y_pred = []
        self.y_score = []
        self.y_pred_proba = y_pred_proba
        self.y_test = y_test
        self.y_true = []
        self.get_predictions()

    def get_predictions(self):
        """
        Gera as predições baseado no valor que está no atributo rule.
        Por fim, ele calcula gera o y_true que é um np.ndarray com as classes corretas.
        """
        match self.rule:
            case "max":
                self.max_rule()
            case "sum":
                self.sum_rule()
            case "mult":
                self.mult_rule()

        self.y_true_no_patch()

    def sum_rule(self):
        pos = []
        sums = []
        for i in range(self.num_groups):
            end_idx, start_idx = self.range_idx(i)
            group_sum = self.y_pred_proba[start_idx:end_idx].sum(axis=0)

            sums.append(group_sum)
            pos.append(np.argmax(group_sum)+1)

        self.y_pred = np.array(pos)
        self.y_score = np.array(sums)

    def range_idx(self, i: int):
        start_idx = i * self.patch
        end_idx = start_idx + self.patch
        return end_idx, start_idx

    def mult_rule(self):
        pos = []
        mults = []
        for i in range(self.num_groups):
            end_idx, start_idx = self.range_idx(i)
            group = self.y_pred_proba[start_idx:end_idx]
            group_mult = group.prod(axis=0)

            mults.append(group_mult)
            pos.append(np.argmax(group_mult)+1)

        self.y_pred = np.array(pos)
        self.y_score = np.array(mults)

    def max_rule(self):
        pos = []
        maxs = []
        for i in range(self.num_groups):
            end_idx, start_idx = self.range_idx(i)
            group = self.y_pred_proba[start_idx:end_idx]
            max_idx = np.argmax(group)
            row_in_group, col = np.unravel_index(max_idx, group.shape)

            pos.append(col+1)
            maxs.append(group[row_in_group])

        self.y_pred = np.array(pos)
        self.y_score = np.array(maxs)

    def y_true_no_patch(self):
        labels = []
        num_groups = self.y_pred_proba.shape[0] // self.patch
        for i in range(num_groups):
            end_idx, start_idx = self.range_idx(i)
            group = self.y_test[start_idx:end_idx]
            labels.append(group[0])

        self.y_true = np.array(labels)


    def save(self, fold, output):
        output_dir = os.path.join(output, self.rule, "predictions")
        os.makedirs(output_dir, exist_ok=True)
        self.save_y_pred(fold, output_dir)
        self.save_y_pred_proba(fold, output_dir)
        self.save_y_score(fold, output_dir)
        self.save_y_true(fold, output_dir)

    def save_y_pred(self, fold, output):
        filename = os.path.join(output, "fold-%d-y_pred-%s.npy" % (fold, self.rule))
        np.save(filename, self.y_pred)

    def save_y_pred_proba(self, fold, output):
        filename = os.path.join(output, "fold-%d-y_pred_proba-%s.npy" % (fold, self.rule))
        np.save(filename, self.y_pred_proba)

    def save_y_score(self, fold, output):
        filename = os.path.join(output, "fold-%d-y_score-%s.npy" % (fold, self.rule))
        np.save(filename, self.y_score)

    def save_y_true(self, fold, output):
        filename = os.path.join(output, "fold-%d-y_true-%s.npy" % (fold, self.rule))
        np.save(filename, self.y_true)
