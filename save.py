import joblib
import logging
import numpy as np
import os
import pandas as pd

from figure import figure_confusion_matrix, figure_topk

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)


def mean_std(list_results, metric):
    return np.mean([result[metric] for result in list_results]), np.std([result[metric] for result in list_results])


def mean_std_topk(results, n_labels):
    k = []
    mean = []
    std = []
    for kk in range(3, n_labels):
        k.append(kk)
        mean.append(np.mean([topk['top_k_accuracy'] for result in results for topk in result['list_topk'] if topk['k'] == kk]))
        std.append(np.std([topk['top_k_accuracy'] for result in results for topk in result['list_topk'] if topk['k'] == kk]))
    return {
        'k': k,
        'mean': mean,
        'std': std
    }


def mean_metrics(list_results, n_labels):
    means = []
    for rule in ['mult', 'sum']:
        results = [result[rule] for result in list_results]
        mean_f1, std_f1 = mean_std(results, 'f1')
        topk = mean_std_topk(results, n_labels)
        means.append({'rule': rule,
                      'mean_f1': mean_f1,
                      'std_f1': std_f1,
                      'topk': topk})
    return means


def save_mean_time(path, results):
    mean_time, std_time = mean_std(results, 'time')
    data = {
        'mean_time': mean_time,
        'std_time': std_time
    }
    filename = os.path.join(path, 'mean_time.csv')
    df = pd.DataFrame(data.values(), index=list(data.keys()))
    save_csv(df, filename, index=True, header=False)


def save_mean_topk(topk, path, rule):
    path_topk = os.path.join(path, 'topk')
    if not os.path.exists(path_topk):
        os.makedirs(path_topk)

    filename = os.path.join(path_topk, 'mean_topk+%s.png' % rule)
    figure_topk(filename, 'Mean Top-$k$', x=topk['k'], y=topk['mean'])

    data = {
        'k': topk['k'],
        'mean': topk['mean'],
        'std': topk['std']
    }
    filename = os.path.join(path_topk, 'mean_topk+%s.csv' % rule)
    df = pd.DataFrame(data)
    save_csv(df, filename, index=False)


def save_mean(means, path, results):
    path_mean = os.path.join(path, 'mean')

    if not os.path.exists(path_mean):
        os.makedirs(path_mean)

    save_mean_time(path_mean, results)

    for rule in ['mult', 'sum']:
        mean = [mean for mean in means if mean['rule'] == rule]
        save_mean_topk(mean[0]['topk'], path_mean, rule)
        data = {
            'mean_f1': mean[0]['mean_f1'],
            'std_f1': mean[0]['std_f1']
        }

        df = pd.DataFrame(data.values(), index=list(data.keys()))
        filename = os.path.join(path_mean, 'mean+%s.csv' % rule)
        save_csv(df, filename, header=False, index=True)


def save_best_mean(means, path):
    mean_mult = [m for m in means if m['rule'] == 'mult']
    mean_sum = [m for m in means if m['rule'] == 'sum']
    best_mean, best_rule = best_mean_and_rule(mean_mult, mean_sum, 'mean_f1')
    data = {
        'best_f1_mean': best_mean,
        'best_f1_rule': best_rule,
    }

    df = pd.DataFrame(data.values(), index=list(data.keys()))
    filename = os.path.join(path, 'best_mean.csv')
    save_csv(df, filename, header=False, index=True)


def save_csv(df, filename, header=True, index=True):
    logging.info('[CSV] %s created' % filename)
    df.to_csv(filename, sep=';', index=index, header=header, lineterminator='\n', doublequote=True)


def save_model_best_classifier(classifier, path):
    filename = os.path.join(path, 'best_model.pkl')
    logging.info('[JOBLIB] %s created' % filename)
    try:
        with open(filename, 'wb') as file:
            joblib.dump(classifier, file, compress=3)
        file.close()
    except FileExistsError:
        logging.warning('problems in save model (%s)' % filename)


def save_best_classifier(classifier, path):
    save_info_best_classifier(classifier, path)
    save_model_best_classifier(classifier, path)


def save_info_best_classifier(classifier, path):
    df = pd.DataFrame(classifier.cv_results_)
    filename = os.path.join(path, 'best_classifier.csv')
    save_csv(df, filename, index=False)


def save_confusion_matrix_csv(list_info_level, path, results):
    for rule in ['sum', 'mult']:
        header = list(list_info_level['levels'].values())
        index = [i[0] + ' (%s)' % i[1] for i in
                 zip(list_info_level['levels'].values(), list_info_level['count'].values())]
        df = pd.DataFrame(results[rule]['confusion_matrix'], index=index, columns=header)
        filename = os.path.join(path, 'cm+%s.csv' % rule)
        save_csv(df, filename)


def save_confusion_matrix(list_info_level, path, results):
    path_confusion_matrix = os.path.join(path, 'confusion_matrix')

    if not os.path.exists(path_confusion_matrix):
        os.makedirs(path_confusion_matrix)

    figure_confusion_matrix('confusion_matrix', list_info_level, path_confusion_matrix, results, title='Confusion Matrix')
    figure_confusion_matrix('confusion_matrix_normalized', list_info_level, path_confusion_matrix, results, title='Confusion Matrix')
    save_confusion_matrix_csv(list_info_level, path_confusion_matrix, results)


def save_topk(list_topk, path, rule):
    path_topk = os.path.join(path, 'topk')

    if not os.path.exists(path_topk):
        os.makedirs(path_topk)

    save_topk_csv(list_topk, path_topk, rule)

    filename = os.path.join(path_topk, 'topk+%s.png' % rule)
    x = [top_k['k'] for top_k in list_topk]
    y = [topk['top_k_accuracy'] for topk in list_topk]
    figure_topk(filename, 'Top-$k$', x, y)


def save_topk_csv(list_topk, path, rule):
    data = {
        'k': [topk['k'] for topk in list_topk],
        'top': [topk['top_k_accuracy'] for topk in list_topk]
    }
    df = pd.DataFrame(data, index=None)
    filename = os.path.join(path, 'topk+%s.csv' % rule)
    save_csv(df, filename, index=False)


def save_fold(fold, path, results):
    for rule in ['mult', 'sum']:
        data = {
            'fold': fold,
            'time': results['time'],
            'f1': results[rule]['f1'],
            'acccuracy': results[rule]['accuracy'],
        }
        
        df = pd.DataFrame(data.values(), index=list(data.keys()))
        filename = os.path.join(path, 'fold+%s.csv' % rule)
        save_csv(df, filename, header=False, index=True)

        save_topk(results[rule]['list_topk'], path, rule)


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
    save_csv(df, filename, header=False, index=True)


def save_best_fold(results, path):
    for rule in ['mult', 'sum']:
        best = max(results, key=lambda x: x[rule]['f1'])
        data = {
            'fold': best['fold'],
            'time': best['time'],
            'f1': best[rule]['f1']
        }

        df = pd.DataFrame(data.values(), index=list(data.keys()))
        filename = os.path.join(path, 'best_fold_%s.csv' % rule)
        save_csv(df, filename, header=False, index=True)


def save_df_main(classifiers, results, path_out):
    extractors = ['mobilenetv2', 'vgg16', 'resnet50v2', 'lbp', 'surf64', 'surf128']
    image_size = [256, 400, 512]
    dimensions = {
        'mobilenetv2': [1280, 1024, 512, 256, 128],
        'vgg16': [512, 256, 128],
        'resnet50v2': [2048, 1024, 512, 256, 128],
        'lbp': [59],
        'surf64': [257, 256, 128],
        'surf128': [513, 512, 256, 128]
    }
    columns = ['%s+%s' % (classifier.__class__.__name__, image) for classifier in classifiers for image in image_size]
    index = ['%s+%s+%s' % (extractor, dimension, metric) for extractor in extractors for dimension in
             dimensions[extractor] for metric in ['mean', 'std']]

    filename = os.path.join(path_out, 'results_final.csv')
    if os.path.exists(filename):
        df = pd.read_csv(filename, names=columns, index_col=0, sep=';')
    else:
        df = pd.DataFrame(index=index, columns=columns)

    for result in results:
        my_column = '%s+%s' % (result['classifier_name'], result['image_size'])
        my_index_mean = '%s+%s+mean' % (result['extractor'], result['n_features'])
        my_index_std = '%s+%s+std' % (result['extractor'], result['n_features'])

        mean_sum = [m for m in result['means'] if m['rule'] == 'sum']
        df[my_column][my_index_mean] = mean_sum[0]['mean_f1']
        df[my_column][my_index_std] = mean_sum[0]['std_f1']

    save_csv(df, filename, header=True, index=True)


def save_best(clf, means, path, results_fold):
    path_best = os.path.join(path, 'best')
    if not os.path.exists(path_best):
        os.makedirs(path_best)
    save_best_classifier(clf, path_best)
    save_best_fold(results_fold, path_best)
    save_best_mean(means, path_best)
