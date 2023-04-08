import logging
import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import ticker

from config import args_topk, args_save, args_confusion_matrix


def figure_confusion_matrix(key, list_info_level, path, results, title, fmt='.2g'):
    args = [a for a in args_confusion_matrix if str(len(list_info_level['levels'])) in a['n_labels']]

    if len(args) == 0:
        raise SystemExit('args for confusion matrix not founded')

    for rule in ['sum', 'mult']:
        cm = results[rule][key]
        filename = os.path.join(path, '%s+%s.png' % (key, rule))
        logging.info('[PNG] confusion matrix %s created' % filename)

        off_diag_mask = np.eye(*cm.shape, dtype=bool)

        figure, axis = plt.subplots(**args[0]['figure'])

        parameters_default = {
            'annot': True,
            'mask': ~off_diag_mask,
            'cmap': 'Reds',
            'fmt': fmt,
            'vmin': np.min(cm),
            'vmax': np.max(cm),
            'ax': axis,
            'annot_kws': {'fontweight': 'bold', 'size': 12}
        }

        axis = sns.heatmap(cm, **parameters_default)
        parameters_default.update({'mask': off_diag_mask, 'cbar': False, 'annot_kws': {}})
        axis = sns.heatmap(cm, **parameters_default)

        posix_xtick = [i + 0.5 for i in range(len(list_info_level['levels'].values()))]
        posix_ytick = [i + 0.5 for i in range(len(list_info_level['levels'].values()))]
        xtick_labels = list_info_level['levels'].values()
        ytick_labels = ['$' + i[0] + '$\n(%s)' % i[1] for i in
                        zip(list_info_level['levels'].values(), list_info_level['count'].values())]

        axis.set_xticks(posix_xtick, labels=xtick_labels, rotation=90,  **args[0]['ticks'])
        axis.xaxis.set_tick_params(pad=16)
        axis.set_yticks(posix_ytick, labels=ytick_labels, rotation=0, **args[0]['ticks'])
        axis.yaxis.set_tick_params(pad=16)
        axis.set_xlabel('True label', **args[0]['label'])
        axis.set_ylabel('Prediction label', **args[0]['label'])
        axis.set_facecolor('white')
        axis.set_title(title, **args[0]['title'])

        plt.ioff()
        plt.tight_layout()
        plt.savefig(filename, **args_save)
        plt.cla()
        plt.clf()
        plt.close(figure)


def figure_topk(filename, title, x, y):
    figure, axis = plt.subplots(**args_topk['figure'])
    plt.plot(x, y, marker='o', color='green')

    axis.set_xlabel('$k$', **args_topk['label'])
    axis.set_ylabel('Correct labels', **args_topk['label'])
    axis.set_facecolor('white')
    axis.set_title(title, **args_topk['title'])

    # Be sure to only pick integer tick locations.
    for axis in [axis.xaxis, axis.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.grid(True)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.25)
    plt.ioff()
    plt.tight_layout()
    plt.savefig(filename, **args_save)
    plt.cla()
    plt.clf()
    plt.close(figure)
