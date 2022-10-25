ROUND_VALUE = 2


def save_best(best_params, list_mean_accuracy, list_mean_f1, list_result_fold, path):

    best_mean_accuracy = max(list_mean_accuracy, key=lambda x: x['mean'])
    best_mean_f1 = max(list_mean_f1, key=lambda x: x['mean'])
    # best_mean_top_k = max(list_mean_top_k, key=lambda x: x['mean'])
    best_fold_accuracy = max(list_result_fold, key=lambda x: x['accuracy'])
    best_fold_f1 = max(list_result_fold, key=lambda x: x['f1_score'])
    best_fold_top_k = max(list_result_fold, key=lambda x: x['max_top_k'])

    values = [[best_params],
              [best_mean_accuracy['rule']],
              [best_mean_accuracy['mean'], round(best_mean_accuracy['mean'] * 100, ROUND_VALUE)],
              [best_fold_accuracy['fold'], best_fold_accuracy['accuracy']],
              [best_mean_f1['rule']],
              [best_mean_f1['mean'], round(best_mean_f1['mean'] * 100, ROUND_VALUE)],
              [best_fold_f1['fold'], best_fold_f1['f1_score']],
              [best_fold_top_k['rule']],
              [best_fold_top_k['fold'], best_fold_top_k['max_top_k']]]
    index = ['best_params',
             'best_mean_accuracy_rule', 'best_mean_accuracy', 'best_fold_accuracy',
             'best_mean_f1_rule', 'best_mean_f1', 'best_fold_f1',
             'best_top_k_rule', 'best_fold_top_k']
    return [{'filename': 'best', 'index': index, 'path': path, 'values': values}]
