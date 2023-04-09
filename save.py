import joblib
import logging
import numpy as np
import os
import pandas as pd

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
    for rule in ['max', 'mult', 'sum']:
        results = [result[rule] for result in list_results]
        mean_f1, std_f1 = mean_std(results, 'f1')
        mean_accuracy, std_accuracy = mean_std(results, 'accuracy')
        topk = mean_std_topk(results, n_labels)
        means.append({'rule': rule,
                      'mean_f1': mean_f1,
                      'std_f1': std_f1,
                      'mean_accuracy': mean_accuracy,
                      'std_accuracy': std_accuracy,
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

    for rule in ['max', 'mult', 'sum']:
        mean = [mean for mean in means if mean['rule'] == rule]
        save_mean_topk(mean[0]['topk'], path_mean, rule)
        for metric in ['f1', 'accuracy']:
            path_metric = os.path.join(path_mean, metric)

            if not os.path.exists(path_metric):
                os.makedirs(path_metric)

            data = {
                'mean_f1': mean[0]['mean_%s' % metric],
                'std_f1': mean[0]['std_%s' % metric]
            }

            df = pd.DataFrame(data.values(), index=list(data.keys()))
            filename = os.path.join(path_metric, 'mean+%s+%s.csv' % (metric, rule))
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


def save_confusion_matrix_csv(confusion_matrix, columns, fname, index, path):
    df = pd.DataFrame(confusion_matrix, index=index, columns=columns)
    filename = os.path.join(path, fname)
    save_csv(df, filename)


def save_confusion_matrix_multilabel(confusion_matrix, count, levels, path, rule):
    path_multilabel = os.path.join(path, 'multilabel')

    if not os.path.exists(path_multilabel):
        os.makedirs(path_multilabel)

    for i, cm in enumerate(confusion_matrix):
        label = levels[i]
        columns = index = ['Positive', 'Negative']
        filename = 'confusion_matrix+%s+%s.csv' % (label, rule)
        save_confusion_matrix_csv(cm, columns, filename, index, path_multilabel)


def save_confusion_matrix(list_info_level, path, results):
    path_confusion_matrix = os.path.join(path, 'confusion_matrix')

    if not os.path.exists(path_confusion_matrix):
        os.makedirs(path_confusion_matrix)

    levels = list_info_level['levels']
    count = list_info_level['count']
    for rule in ['max', 'mult', 'sum']:
        for type_confusion_matrix in ['confusion_matrix', 'confusion_matrix_normalized', 'confusion_matrix_multilabel']:
            confusion_matrix = results[rule][type_confusion_matrix]
            if type_confusion_matrix == 'confusion_matrix_multilabel':
                save_confusion_matrix_multilabel(confusion_matrix, count, levels, path_confusion_matrix, rule)
            else:
                save_confusion_matrix_and_normalized(confusion_matrix, count, levels, list_info_level,
                                                     path_confusion_matrix, rule, type_confusion_matrix)


def save_confusion_matrix_and_normalized(confusion_matrix, count, levels, list_info_level, path_confusion_matrix, rule,
                                         type_confusion_matrix):
    columns = list(list_info_level['levels'].values())
    index = [i[0] + ' (%s)' % i[1] for i in zip(levels.values(), count.values())]
    filename = '%s+%s.csv' % (type_confusion_matrix, rule)
    save_confusion_matrix_csv(confusion_matrix, columns, filename, index, path_confusion_matrix)


def save_topk(list_topk, path, rule):
    path_topk = os.path.join(path, 'topk')

    if not os.path.exists(path_topk):
        os.makedirs(path_topk)

    save_topk_csv(list_topk, path_topk, rule)


def save_topk_csv(list_topk, path, rule):
    data = {
        'k': [topk['k'] for topk in list_topk],
        'top': [topk['top_k_accuracy'] for topk in list_topk]
    }
    df = pd.DataFrame(data, index=None)
    filename = os.path.join(path, 'topk+%s.csv' % rule)
    save_csv(df, filename, index=False)


def save_classification_report(classification_report, path, rule):
    path_classification_report = os.path.join(path, 'classification_report')

    if not os.path.exists(path_classification_report):
        os.makedirs(path_classification_report)

    df = pd.DataFrame(classification_report).transpose()
    filename = os.path.join(path_classification_report, 'classification_report+%s.csv' % rule)
    save_csv(df, filename)


def save_fold(count_train, count_test, fold, path, results):
    for rule in ['max', 'mult', 'sum']:
        data = {
            'fold': fold,
            'time': results['time'],
            'f1': results[rule]['f1'],
            'acccuracy': results[rule]['accuracy'],
            'count_train': count_train,
            'count_test': count_test
        }
        
        df = pd.DataFrame(data.values(), index=list(data.keys()))
        filename = os.path.join(path, 'fold+%s.csv' % rule)
        save_csv(df, filename, header=False, index=True)

        save_topk(results[rule]['list_topk'], path, rule)
        save_classification_report(results[rule]['classification_report'], path, rule)


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
    for rule in ['max', 'mult', 'sum']:
        best = max(results, key=lambda x: x[rule]['f1'])
        data = {
            'fold': best['fold'],
            'time': best['time'],
            'f1': best[rule]['f1']
        }

        df = pd.DataFrame(data.values(), index=list(data.keys()))
        filename = os.path.join(path, 'best_fold_%s.csv' % rule)
        save_csv(df, filename, header=False, index=True)


def save_df_main(dataset_name, dimensions, minimum_image, results, path):
    extractors = ['mobilenetv2', 'vgg16', 'resnet50v2', 'lbp', 'surf64', 'surf128']
    image_size = [256, 400, 512]
    classifiers_name = [
        'KNeighborsClassifier',
        'MLPClassifier',
        'RandomForestClassifier',
        'SVC',
        'DecisionTreeClassifier',
    ]
    columns = ['%s+%s' % (name, image) for name in classifiers_name for image in image_size]
    index = ['%s+%s+%s' % (extractor, dimension, metric) for extractor in extractors for dimension in
             dimensions[extractor] for metric in ['mean', 'std']]

    filename = os.path.join(path, '%s+results_final+%s.csv' % (dataset_name, str(minimum_image)))
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
