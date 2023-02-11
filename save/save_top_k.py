import itertools
import os
import pathlib
import shutil

import numpy as np
import pandas as pd

import matplotlib.style as mplstyle
from matplotlib import pyplot as plt
from matplotlib import ticker

from save.save_files import save_df, compress_folder

mplstyle.use(['ggplot', 'fast'])


def save_info_top_k(index, path, rule, values):
    df = pd.DataFrame(values, index)
    filename = 'info_top_k_%s' % rule
    save_df(df, filename, path, header=False)


def save_top_k(path, rule, values):
    df = pd.DataFrame(values)
    filename = 'top_k_%s' % rule
    save_df(df, filename, path, index=False)


def save_top_k_rule(path_fold, result, rule):
    if len(result[0]['top_k']) > 0:
        fold, max_top_k, min_top_k, top_k, y_test = get_info_top_k(result[0])
        index = ['rule', 'min_top_k', 'max_top_k', 'total']
        values = [rule, min_top_k, max_top_k, len(y_test)]

        path_top_k = os.path.join(path_fold, 'top_k', rule)
        pathlib.Path(path_top_k).mkdir(exist_ok=True, parents=True)

        save_plot_top_k(fold, max_top_k, min_top_k, path_top_k, rule, top_k, y_test)

        save_info_top_k(index, path_top_k, rule, values)
        save_top_k(path_top_k, rule, top_k)

        filename_compress = os.path.join(path_fold, 'top_k.tar.gz')
        foldername = os.path.join(path_fold, 'top_k')
        compress_folder(filename_compress, foldername)
        print('[TOP-K] compress folder %s' % foldername)

        shutil.rmtree(foldername)
        print('[TOP-K] delete folder %s' % foldername)


def get_info_top_k(result):
    top_k = result['top_k']
    fold = result['fold']
    max_top_k = result['max_top_k']
    min_top_k = result['min_top_k']
    y_test = result['y_true']
    return fold, max_top_k, min_top_k, top_k, y_test


def save_plot_top_k(fold, max_top_k, min_top_k, path_fold, rule, top_k, y_true):
    result_top_k = save_top_k_three_and_five(fold, max_top_k, min_top_k, path_fold, rule, top_k, y_true)

    filename = os.path.join(path_fold, 'top_k_%s.png' % rule)
    title = 'All top $k$'
    plot_top_k(filename, 'top_k_accuracy', result_top_k, max_top_k, min_top_k, title, y_true)


def save_top_k_three_and_five(fold, max_top_k, min_top_k, path_fold, rule, top_k, y_true):
    for k in [3, 5]:
        result_top_k = [x for x in top_k if x['k'] <= k]
        filename = os.path.join(path_fold, f'top_k_{rule}_k={k}.png')
        title = f'Top $k$\nRule: %s, $k$: %d, Fold: %d,\n' % (rule, k, fold)
        plot_top_k(filename, 'top_k_accuracy', result_top_k, max_top_k, min_top_k, title, y_true)
    return result_top_k


def plot_top_k(filename, key, list_top_k, max_top_k, min_top_k, title, y_test):
    x = [top_k['k'] for top_k in list_top_k]
    y = [top_k[key] for top_k in list_top_k]

    title = title + f'Minimum value top $k$: %d,\nMaximum value top $k$: %d,\nCount of tests: %d' % (min_top_k, max_top_k, len(y_test))
    fontsize_title = 14
    pad_title = 20
    fontsize_label = 14

    plot_size = (10, 10)

    figure, axis = plt.subplots(figsize=plot_size)
    plt.plot(x, y, marker='o', color='green')

    axis.set_title(title, fontsize=fontsize_title, pad=pad_title)
    axis.set_xlabel('$k$', fontsize=fontsize_label)
    axis.set_ylabel('Count of labels in top $k$', fontsize=fontsize_label)

    plt.grid(True)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.25)

    # Be sure to only pick integer tick locations.
    for axis in [axis.xaxis, axis.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.ioff()
    plt.rcParams['figure.facecolor'] = 'white'

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f'[TOP-K] save {filename}')
    plt.cla()
    plt.clf()
    plt.close(figure)


def mean_top_k(list_result_fold, path):
    p = os.path.join(path, 'mean_top_k')
    pathlib.Path(p).mkdir(exist_ok=True, parents=True)

    mean_top_k_each_rule(list_result_fold, p)


def mean_top_k_each_rule(list_result_fold, path):
    for rule in ['max', 'prod', 'sum']:
        list_top_k = [x['top_k'] for x in list_result_fold if x['rule'] == rule]
        list_top_k = list(itertools.chain.from_iterable(list_top_k))
        if len(list_top_k) > 0:
            min_k = min(list_top_k, key=lambda x: x['k'])['k']
            max_k = max(list_top_k, key=lambda x: x['k'])['k']

            list_each_k = []
            for i in range(min_k, max_k + 1):
                values_k = [k['top_k_accuracy'] for k in list_top_k if k['k'] == i]
                list_each_k.append({'k': i, 'values': values_k, 'top_k': np.mean(values_k)})

            df = pd.DataFrame(list_each_k)
            filename = 'mean_top_%s' % rule
            save_df(df, filename, path, index=False)

            title = 'Mean of top $k$\n'
            filename = os.path.join(path, 'mean_top_k_%s.png' % rule)
            min_top_k = min(list_each_k, key=lambda x: x['top_k'])['top_k']
            max_top_k = max(list_each_k, key=lambda x: x['top_k'])['top_k']
            plot_top_k(filename, 'top_k', list_each_k, min_top_k, max_top_k, title, list_each_k)
