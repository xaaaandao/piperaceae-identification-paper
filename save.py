import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


def mean_std(list_results, metric):
    return np.mean([result[metric] for result in list_results]), np.std([result[metric] for result in list_results])


def mean_topk(results):
    mean_topk_three, std_topk_three = mean_std(results, 'topk_three')
    mean_topk_five, std_topk_five = mean_std(results, 'topk_five')
    return mean_topk_five, mean_topk_three, std_topk_five, std_topk_three


def mean_metrics(list_results):
    means = []
    for rule in ['mult', 'sum']:
        results = [result[rule] for result in list_results]
        mean_f1, std_f1 = mean_std(results, 'f1')
        mean_topk_five, mean_topk_three, std_topk_five, std_topk_three = mean_topk(results)
        means.append({'rule': rule,
                          'mean_f1': mean_f1,
                          'std_f1': std_f1,
                          'mean_topk_three': mean_topk_three,
                          'std_topk_three': std_topk_three,
                          'mean_topk_five': mean_topk_five,
                          'std_topk_five': std_topk_five})
    return means


def save_mean(means, path):
    rules = sorted(['sum', 'mult'])
    data = {
        'rule': [m['rule'] for m in means for rule in rules if m['rule'] == rule],
        'mean_f1': [m['rule'] for m in means for rule in rules if m['rule'] == rule],
        'std_f1': [m['std_f1'] for m in means for rule in rules if m['rule'] == rule],
        'mean_topk_three': [m['mean_topk_three'] for m in means for rule in rules if m['rule'] == rule],
        'std_topk_three': [m['std_topk_three'] for m in means for rule in rules if m['rule'] == rule],
        'mean_topk_five': [m['mean_topk_five'] for m in means for rule in rules if m['rule'] == rule],
        'std_topk_five': [m['std_topk_five'] for m in means for rule in rules if m['rule'] == rule]
    }
    df = pd.DataFrame(data.values(), index=data.keys())
    filename = os.path.join(path, 'mean.csv')
    save_csv(df, filename, header=False, index=data.keys())


def save_best_mean(means, path):
    mean_mult = [m for m in means if m['rule'] == 'mult']
    mean_sum = [m for m in means if m['rule'] == 'sum']

    best_f1_mean, best_f1_rule = best_mean_and_rule(mean_mult, mean_sum, 'mean_f1')
    best_topk_three_mean, best_topk_three_rule = best_mean_and_rule(mean_mult, mean_sum, 'mean_topk_three')
    best_topk_five_mean, best_topk_five_rule = best_mean_and_rule(mean_mult, mean_sum, 'mean_topk_five')

    data = [best_f1_mean, best_f1_rule, best_topk_three_mean, best_topk_three_rule, best_topk_five_mean, best_topk_five_rule]
    index = ['best_f1_mean', 'best_f1_rule', 'best_topk_three_mean', 'best_topk_three_rule', 'best_topk_five_mean', 'best_topk_five_rule']

    df = pd.DataFrame(data, index=index)
    filename = os.path.join(path, 'best_mean.csv')
    save_csv(df, filename, header=False, index=index)


def save_csv(df, filename, header=True, index=True):
    print('[CSV] %s created' % filename)
    df.to_csv(filename, sep=';', index=index, header=header, lineterminator='\n', doublequote=True)


def save_model_best_classifier(classifier, path):
    filename = os.path.join(path, 'model.pkl')
    print('[JOBLIB] %s created' % filename)
    try:
        with open(filename, 'wb') as file:
            joblib.dump(classifier, file, compress=3)
        file.close()
    except FileExistsError:
        print('problems in save model (%s)' % filename)


def save_best_classifier(classifier, path):
    save_info_best_classifier(classifier, path)
    save_model_best_classifier(classifier, path)


def save_info_best_classifier(classifier, path):
    df = pd.DataFrame(classifier.cv_results_)
    filename = os.path.join(path, 'best_classifier.csv')
    save_csv(df, filename, index=False)


def save_cm_figure(list_info_level, path, results):
    for rule in ['sum', 'mult']:
        cm = results[rule]['confusion_matrix']
        figsize = (25, 25)
        fig, axis = plt.subplots(figsize=figsize)
        filename = os.path.join(path, 'cm+%s.png' % rule)
        print('[PNG] confusion matrix %s created' % filename)

        off_diag_mask = np.eye(*cm.shape, dtype=bool)

        figure, axis = plt.subplots(figsize=(10, 10))

        parameters = {
            'annot': True,
            'mask': ~off_diag_mask,
            'cmap': 'Reds',
            'fmt': '.2g',
            'vmin': np.min(cm),
            'vmax': np.max(cm),
            'ax': axis,
        }
        axis = sns.heatmap(cm, **parameters)

        parameters.update({'mask': off_diag_mask, 'cbar': False, 'annot_kws': {}})
        axis = sns.heatmap(cm, **parameters)

        posix_xtick = [i + 0.5 for i in range(len(list_info_level['levels'].values()))]
        posix_ytick = [i + 0.5 for i in range(len(list_info_level['levels'].values()))]
        xtick_labels = list_info_level['levels'].values()
        ytick_labels = [i[0] + ' (%s)' % i[1] for i in zip(list_info_level['levels'].values(), list_info_level['count'].values())]

        axis.set_xticks(posix_xtick, labels=xtick_labels, fontsize=8, rotation=90)
        axis.set_yticks(posix_ytick, labels=ytick_labels, fontsize=8, rotation=0)
        axis.set_xlabel('True label', fontsize=14)
        axis.set_ylabel('Prediction label', fontsize=14)
        axis.set_facecolor('white')
        axis.set_title('test', fontsize=24, pad=32)

        plt.ioff()
        plt.tight_layout()
        plt.savefig(filename, format='png', dpi=300)
        plt.cla()
        plt.clf()
        plt.close(fig)


def save_cm_csv(list_info_level, path, results):
    for rule in ['sum', 'mult']:
        header = list(list_info_level['levels'].values())
        index = [i[0] + ' (%s)' % i[1] for i in zip(list_info_level['levels'].values(), list_info_level['count'].values())]
        df = pd.DataFrame(results[rule]['confusion_matrix'], index=index, columns=header)
        filename = os.path.join(path, 'cm+%s.csv' % rule)
        save_csv(df, filename)


def save_confusion_matrix(list_info_level, path, results):
    save_cm_figure(list_info_level, path, results)
    save_cm_csv(list_info_level, path, results)


def save_fold(fold, path, results):
    index = ['fold', 'results_f1_mult', 'results_f1_sum', 'results_topk_mult', 'results_topk_sum']
    data = [fold, results['mult']['f1'], results['sum']['f1'], results['mult']['topk_three'], results['sum']['topk_five']]
    df = pd.DataFrame(data, index=index)
    filename = os.path.join(path, 'fold.csv')
    save_csv(df, filename, header=False, index=index)


def best_mean_and_rule(mean_mult, mean_sum, metric):
    if len(mean_mult) == 0 or len(mean_sum) == 0:
        raise SystemError('mean not found')

    best_mean = mean_sum[0][metric] if mean_sum[0][metric] > mean_mult[0][metric] else mean_mult[0][metric]
    best_rule = 'sum' if mean_sum[0][metric] > mean_mult[0][metric] else 'mult'
    return best_mean, best_rule


def save_info(classifier_name, extractor, n_features, n_samples, path, patch):
    index = ['classifier_name', 'extractor', 'n_features', 'n_samples', 'path', 'patch']
    data = [classifier_name, extractor, n_features, n_samples, path, patch]
    df = pd.DataFrame(data, index=index)
    filename = os.path.join(path, 'info.csv')
    save_csv(df, filename, header=False, index=index)


def save_best_fold(results, path):
    data = []
    index = []
    for rule in ['mult', 'sum']:
        for metric in ['f1', 'topk_three', 'topk_five']:
            best = max(results, key=lambda x: x[rule][metric])
            data.append(best['fold'])
            data.append(best[rule][metric])
            index.append('best_%s_%s_fold' % (metric, rule))
            index.append('best_%s_%s' % (metric, rule))

    df = pd.DataFrame(data, index=index)
    filename = os.path.join(path, 'best_fold.csv')
    save_csv(df, filename, header=False, index=index)
