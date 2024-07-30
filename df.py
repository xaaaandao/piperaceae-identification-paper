import itertools
import logging
import os
import pathlib
from typing import LiteralString

import numpy as np
import pandas as pd


class DF:
    def __init__(self, count_train, count_test, dataset, final_time, predicts, total_train, total_test,
                 total_train_no_patch, total_test_no_patch):
        self.count_train = count_train
        self.count_test = count_test
        self.dataset = dataset
        self.final_time = final_time
        self.predicts = predicts
        self.total_train = total_train
        self.total_test = total_test
        self.total_train_no_patch = total_train_no_patch
        self.total_test_no_patch = total_test_no_patch
        self.info = self.create_info()
        self.count_train_test = self.create_count_train_test(dataset)
        self.evals = self.create_evaluations()
        self.best = self.create_best(self.evals)
        self.preds = self.create_predicts(dataset.levels)
        self.top = self.create_top()
        self.classifications = self.create_classification()
        self.confusion_matrix = self.create_confusion_matrix()
        self.confusion_matrix_normalized = self.create_confusion_matrix_normalized()
        self.confusion_matrix_multilabel = self.create_confusion_matrix_multilabel()
        self.true_positive = self.create_true_positive(dataset)

    def create_evaluations(self):
        data = {
            'accuracy': [p.accuracy for p in sorted(self.predicts, key=lambda x: x.rule)],
            'f1': [p.f1 for p in sorted(self.predicts, key=lambda x: x.rule)],
            'rule': [p.rule for p in sorted(self.predicts, key=lambda x: x.rule)]
        }
        return pd.DataFrame(data, index=None, columns=list(data.keys()))

    def create_count_train_test(self, dataset):
        data = {'labels': [], 'trains': [], 'tests': []}
        for train, test in zip(sorted(self.count_train.items()), sorted(self.count_test.items())):
            data['trains'].append(train[1] / dataset.image.patch)
            data['tests'].append(test[1] / dataset.image.patch)
            data['labels'].append(train[0])

        df = pd.DataFrame(data, index=None, columns=list(data.keys()))
        df['labels'] = df[['labels']].map(
            lambda row: list(filter(lambda x: x.label.__eq__(row), dataset.levels))[0].specific_epithet)
        return df

    def create_info(self):
        data = {
            'time': [self.final_time],
            'total_test': [self.total_test],
            'total_train': [self.total_train],
            'total_test_no_patch': [self.total_test_no_patch],
            'total_train_no_patch': [self.total_train_no_patch],
        }
        return pd.DataFrame(data, index=None, columns=list(data.keys()))

    def create_predicts(self, levels):
        data = {
            'y_pred+sum': list(itertools.chain(*[p.y_pred.tolist() for p in self.predicts if p.rule.__eq__('sum')])),
            'y_pred+mult': list(itertools.chain(*[p.y_pred.tolist() for p in self.predicts if p.rule.__eq__('mult')])),
            'y_pred+max': list(itertools.chain(*[p.y_pred.tolist() for p in self.predicts if p.rule.__eq__('max')])),
            'y_true': list(itertools.chain(*[self.predicts[0].y_true.tolist()]))
        }
        df = pd.DataFrame(data, index=None, columns=list(data.keys()))

        if len(levels) > 0:
            df = df.map(lambda row: list(filter(lambda x: x.label.__eq__(row), levels))[0].specific_epithet)

        df['equals'] = df.apply(lambda row: row[row == row['y_true']].index.tolist(), axis=1)
        return df

    def set_top(self, predict):
        data = {
            'k': [topk.k for topk in sorted(predict.topk, key=lambda x: x.k)],
            'topk_accuracy_score': [topk.top_k_accuracy_score for topk in sorted(predict.topk, key=lambda x: x.k)],
            'count_test': np.repeat(self.total_test_no_patch, len(predict.topk)),
            'topk_accuracy_score+100': [topk.top_k_accuracy_score / self.total_test_no_patch for topk in
                                        sorted(predict.topk, key=lambda x: x.k)],
            'rule': [predict.rule] * len(predict.topk)  # equivalent a np.repeat, but works in List[str]
        }
        return pd.DataFrame(data, columns=list(data.keys()), index=None)

    def create_top(self):
        data = {'k': [], 'topk_accuracy_score': [], 'rule': []}
        df = pd.DataFrame(data, columns=list(data.keys()))
        for predict in self.predicts:
            top = self.set_top(predict)
            df = pd.concat([df, top], axis=0)
        return df

    def create_best(self, df_evaluations):
        df_accuracy = df_evaluations.loc[df_evaluations['accuracy'].idxmax()]
        df_f1 = df_evaluations.loc[df_evaluations['f1'].idxmax()]
        data = {'metric': ['accuracy', 'f1'],
                'value': [df_accuracy['accuracy'], df_f1['f1']],
                'rule': [df_accuracy['rule'], df_f1['rule']]}
        return pd.DataFrame(data, columns=list(data.keys()), index=None)

    def set_true_positive(self, dataset, predict):
        data = {
            'labels': self.get_level(dataset),
            'count_train': self.get_count_train(dataset),
            'count_test': self.get_count_test(dataset),
            'true_positive': list(np.diag(predict.confusion_matrix)),
            'rule': [predict.rule] * len(dataset.levels)
        }
        return pd.DataFrame(data, columns=list(data.keys()))

    def create_true_positive(self, dataset):
        data = {
            'labels': [],
            'true_positive': [],
            'rule': []
        }
        df = pd.DataFrame(data, columns=list(data.keys()))
        for predict in self.predicts:
            best = self.set_true_positive(dataset, predict)
            df = pd.concat([df, best], axis=0)
        return df

    def get_count(self, count: dict, label: int):
        """
        Encontra a quantidade de treinos de uma determinada classe.
        :param count: coleção com todas as quantidades de treinos.
        :param label: classe que deseja encontrar.
        :return: quantidade de treinos de uma determinada classe.
        """
        for count in list(count.items()):
            if label == count[0]:
                return count[1]

    def get_level(self, dataset):
        """
        Cria uma lista com os levels (classes), ou seja, nome e a quantidade de treinos e testes, que serão utilizados na matriz de confusão.
        :param levels: levels (classes) com nome das espécies utilizadas.
        :param patch: quantidade de divisões da imagem.
        :return: lista com o nome das classes e a quantidade de treinos de testes.
        """
        return [level.specific_epithet for level in sorted(dataset.levels, key=lambda x: x.label)]

    def get_count_train(self, dataset):
        return [int(self.get_count(self.count_train, level.label) / dataset.image.patch)
                for level in sorted(dataset.levels, key=lambda x: x.label)]

    def get_count_test(self, dataset):
        return [int(self.get_count(self.count_test, level.label) / dataset.image.patch)
                for level in sorted(dataset.levels, key=lambda x: x.label)]

    def save_info(self, output):
        filename = os.path.join(output, 'info.csv')
        self.info.to_csv(filename, index=False, header=True, sep=';', quoting=2, encoding='utf-8')

    def save_evals(self, output):
        filename = os.path.join(output, 'evals.csv')
        self.evals.to_csv(filename, index=False, header=True, sep=';', quoting=2, encoding='utf-8')

    def save_count_train_test(self, output):
        filename = os.path.join(output, 'count_train_test.csv')
        self.count_train_test.to_csv(filename, index=False, header=True, sep=';', quoting=2, encoding='utf-8')

    def save_preds(self, output):
        filename = os.path.join(output, 'preds.csv')
        self.preds.to_csv(filename, index=False, header=True, sep=';', quoting=2, encoding='utf-8')

    def save_true_positive(self, output):
        filename = os.path.join(output, 'true_positive.csv')
        self.true_positive.to_csv(filename, index=False, header=True, sep=';', quoting=2, encoding='utf-8')

    def save_top(self, output):
        filename = os.path.join(output, 'top.csv')
        self.top.to_csv(filename, index=False, header=True, sep=';', quoting=2, encoding='utf-8')

    def save_best(self, output):
        filename = os.path.join(output, 'best+evals.csv')
        self.best.to_csv(filename, index=False, header=True, sep=';', quoting=2, encoding='utf-8')

    def save(self, output):
        self.save_info(output)
        self.save_count_train_test(output)
        self.save_evals(output)
        self.save_best(output)
        self.save_preds(output)
        self.save_top(output)
        self.save_true_positive(output)
        self.save_classifications(output)
        columns = self.get_columns(self.dataset.levels)
        index = self.get_index(self.count_train, self.count_test, self.dataset.levels, self.dataset.image.patch)
        self.save_confusion_matrix(columns, index, output)
        self.save_confusion_matrix_normalized(columns, index, output)
        self.save_confusion_matrix_multilabel(self.dataset.levels, output)

    def create_classification(self):
        return {p.rule: p.classification_report for p in self.predicts}

    def save_classifications(self, output):
        output = os.path.join(output, 'classification_report')
        os.makedirs(output, exist_ok=True)
        for k, v in self.classifications.items():
            filename = os.path.join(output, 'classification_report+%s.csv' % k)
            df = pd.DataFrame(v).transpose()
            df.to_csv(filename, index=True, header=True, sep=';', quoting=2)
            logging.info('Saving %s' % filename)

    def get_index(self, count_train: dict, count_test: dict, levels: list, patch: int):
        """
        Cria uma lista com os levels (classes), ou seja, nome e a quantidade de treinos e testes, que serão utilizados na matriz de confusão.
        :param count_train: coleção com a quantidade de treinos da matriz de confusão.
        :param count_test: coleção com a quantidade de testes da matriz de confusão.
        :param levels: levels (classes) com nome das espécies utilizadas.
        :param patch: quantidade de divisões da imagem.
        :return: lista com o nome das classes e a quantidade de treinos de testes.
        """
        return [level.specific_epithet + '(%d-%d)'
                % (int(self.get_count_train_label(count_train, level.label) / patch),
                   int(self.get_count_test_label(count_test, level.label) / patch))
                for level in sorted(levels, key=lambda x: x.label)]

    def save_confusion_matrix(self, columns, index, output):
        output = os.path.join(output, 'confusion_matrix')
        os.makedirs(output, exist_ok=True)
        for k, v in self.confusion_matrix.items():
            filename = os.path.join(output, 'confusion_matrix+%s.csv' % k)
            df = pd.DataFrame(v, index=index, columns=columns)
            df.to_csv(filename, index=index, header=True, sep=';', quoting=2)
            logging.info('Saving %s' % filename)

    def save_confusion_matrix_normalized(self, columns, index, output):
        output = os.path.join(output, 'confusion_matrix', 'normalized')
        os.makedirs(output, exist_ok=True)
        for k, v in self.create_confusion_matrix_normalized().items():
            filename = os.path.join(output, 'confusion_matrix+%s.csv' % k)
            df = pd.DataFrame(v, index=index, columns=columns)
            df.to_csv(filename, index=index, header=True, sep=';', quoting=2)
            logging.info('Saving %s' % filename)

    def get_count_train_label(self, count_train: dict, label: int):
        """
        Encontra a quantidade de treinos de uma determinada classe.
        :param count_train: coleção com todas as quantidades de treinos.
        :param label: classe que deseja encontrar.
        :return: quantidade de treinos de uma determinada classe.
        """
        for count in list(count_train.items()):
            if label == count[0]:
                return count[1]

    def get_count_test_label(self, count_test: dict, label: int):
        """
        Encontra a quantidade de testes de uma determinada classe.
        :param count_test: coleção com todas as quantidades de testes.
        :param label: classe que deseja encontrar.
        :return: quantidade de treinos de uma determinada classe.
        """
        for count in list(count_test.items()):
            if label == count[0]:
                return count[1]

    def get_columns(self, levels):
        """
        Cria uma lista com os levels (classes) que serão utilizados na matriz de confusão.
        :param levels: levels (classes) com nome das espécies utilizadas.
        :return: lista com o nome das classes.
        """
        return [level.specific_epithet for level in levels]

    def create_confusion_matrix(self):
        return {p.rule: p.confusion_matrix for p in self.predicts}

    def create_confusion_matrix_normalized(self):
        return {p.rule: p.confusion_matrix_normalized for p in self.predicts}

    def create_confusion_matrix_multilabel(self):
        return {p.rule: p.confusion_matrix_multilabel for p in self.predicts}


    def save_confusion_matrix_multilabel(self, levels: list, output: pathlib.Path | LiteralString | str):
        """
        Salva a matriz de confusão de uma classe em um arquivo CSV.
        :param levels: lista com os levels (classes) da matriz de confusão.
        :param output: local aonde será salvo a matriz de confusão.
        :param rule: regra que foi utilizada naquela matriz de confusão.
        """
        columns = index = ['Positive', 'Negative']
        p = os.path.join(output, 'confusion_matrix', 'multilabel')
        for k, v in self.confusion_matrix_multilabel.items():
            path_full = os.path.join(p, k)
            os.makedirs(path_full, exist_ok=True)
            for idx, confusion_matrix in enumerate(v, start=1):
                level = list(filter(lambda x: x.label.__eq__(idx), levels))
                if len(level) < 1:
                    raise ValueError

                filename = os.path.join(path_full, 'confusion_matrix_multilabel=%s.csv' % level[0].specific_epithet)
                df = pd.DataFrame(confusion_matrix, index=index, columns=columns)
                df.to_csv(filename, index=index, header=True, sep=';', quoting=2)
                logging.info('Saving %s' % filename)