# import os
# import pathlib
# import time
#
# import numpy as np
# import sklearn.preprocessing
#
# from classifier import find_best_classifier_and_params, list_classifiers
# from data import get_info, data_with_pca
# from result import calculate_test
# from save import save
#
#
# def handcraft(cfg, current_datetime, kf, list_data_input, list_extractor):
#     n_patch = None
#     patch_slice = None
#
#     list_only_file = [file for file in list_data_input if os.path.isfile(file)]
#
#     for file in list_only_file:
#         dataset, color_mode, segmented, dim, extractor = get_info(file)
#         print(f'dataset:{dataset}\ncolor_mode:{color_mode}\nsegmented:{segmented}\ndim:{dim}\nextractor:{extractor}')
#
#         data = np.loadtxt(file)
#         n_samples, n_features = data.shape
#         x, y = data[0:, 0:n_features - 1], data[:, n_features - 1]
#         n_labels = len(np.unique(y))
#
#         if not np.isnan(x).any():
#
#             x_normalized = sklearn.preprocessing.StandardScaler().fit_transform(x)
#
#             list_data_pca = data_with_pca(cfg, extractor, list_extractor, x_normalized, y)
#
#             for data in list_data_pca:
#                 for classifier in list_classifiers:
#                     classifier_name = classifier.__class__.__name__
#
#                     best, time_search_best_params = find_best_classifier_and_params(
#                         cfg,
#                         classifier,
#                         classifier_name,
#                         data)
#
#                     list_result_fold = []
#                     list_time = []
#
#                     path = os.path.join(cfg['dir_output'], current_datetime, dataset, segmented, color_mode, dim,
#                                         extractor, classifier_name,
#                                         'patch=' + str(n_patch),
#                                         str(data['pca']))
#                     pathlib.Path(path).mkdir(parents=True, exist_ok=True)
#
#                     for fold, (index_train, index_test) in enumerate(kf.split(np.random.rand(n_samples, ))):
#                         x_train, y_train = x[index_train], y[index_train]
#                         x_test, y_test = x[index_test], y[index_test]
#
#                         start_time_train_valid = time.time()
#                         best['classifier'].fit(x_train, y_train)
#                         y_pred = best['classifier'].predict_proba(x_test)
#                         result_max_rule, result_prod_rule, result_sum_rule = calculate_test(fold, n_labels, y_pred, y_test)
#                         end_time_train_valid = time.time()
#                         time_train_valid = end_time_train_valid - start_time_train_valid
#
#                         list_result_fold.append(result_max_rule)
#                         list_result_fold.append(result_prod_rule)
#                         list_result_fold.append(result_sum_rule)
#                         list_time.append({
#                             "fold": fold,
#                             "time_train_valid": time_train_valid,
#                             "time_search_best_params": time_search_best_params
#                         })
#
#                     save(best['params'], cfg, classifier_name, color_mode, data, dataset, dim, extractor, file,
#                          list_result_fold, list_time, n_patch, path, patch_slice)
#         #             break
#         #         break
#         #     break
#         # break