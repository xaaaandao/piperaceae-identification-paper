import numpy as np

ROUND_VALUE = 2


def save_mean(list_result_fold, list_time, path):
    mean_time_train_valid, std_time_train_valid = get_mean_std_time('time_train_valid', list_time)
    mean_time_search_best_params, std_time_search_best_params = get_mean_std_time('time_search_best_params', list_time)

    list_mean_accuracy = get_list_all_rule(list_result_fold, 'accuracy')
    list_mean_f1 = get_list_all_rule(list_result_fold, 'f1_score')

    accuracy = get_mean_metric(list_mean_accuracy)
    f1 = get_mean_metric(list_mean_f1)

    values = [mean_time_train_valid, std_time_train_valid,
              mean_time_search_best_params, std_time_search_best_params] + accuracy + f1

    index = ['mean_time_train_valid', 'std_time_train_valid',
             'mean_time_search_best_params', 'std_time_search_best_params',
             'mean_accuracy_max', 'std_accuracy_max',
             'mean_accuracy_sum', 'std_accuracy_sum',
             'mean_accuracy_prod', 'std_accuracy_prod',
             'mean_f1_max', 'std_f1_max',
             'mean_f1_sum', 'std_f1_sum',
             'mean_f1_prod', 'std_f1_prod']

    return list_mean_accuracy, list_mean_f1, \
           [{'filename': 'mean', 'index': index, 'path': path, 'values': values}]


def get_mean_metric(list_mean_metric):
    metric_max = list(filter(lambda x: x['rule'] == 'max', list_mean_metric))
    mean_metric_max = metric_max[0]['mean']
    std_metric_max = metric_max[0]['std']

    metric_sum = list(filter(lambda x: x['rule'] == 'sum', list_mean_metric))
    mean_metric_sum = metric_sum[0]['mean']
    std_metric_sum = metric_sum[0]['std']

    metric_prod = list(filter(lambda x: x['rule'] == 'prod', list_mean_metric))
    mean_metric_prod = metric_prod[0]['mean']
    std_metric_prod = metric_prod[0]['std']

    return [[mean_metric_max], [std_metric_max],
            [mean_metric_sum], [std_metric_sum],
            [mean_metric_prod], [std_metric_prod]]


def get_list_all_rule(list_result_fold, metric):
    return [calculate_mean_by_metric_and_rule(list_result_fold, metric, 'sum'),
            calculate_mean_by_metric_and_rule(list_result_fold, metric, 'max'),
            calculate_mean_by_metric_and_rule(list_result_fold, metric, 'prod')]


def calculate_mean_by_metric_and_rule(list_result_fold, metric, rule):
    return {
        'mean': np.mean([r[metric] for r in list(filter(lambda x: x['rule'] == rule, list_result_fold))]),
        'std': np.std([r[metric] for r in list(filter(lambda x: x['rule'] == rule, list_result_fold))]),
        'rule': rule,
        'metric': metric
    }


def get_mean_std_time(key, list_time):
    mean_time = np.mean([time[key] for time in list_time])
    round_mean_time = round(float(mean_time), ROUND_VALUE)

    std_time = np.std([time[key] for time in list_time])
    round_std_time = round(float(std_time), ROUND_VALUE)

    return [mean_time, round_mean_time], [std_time, round_std_time]
