import logging
import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import ticker


def figure_confusion_matrix(list_info_level, path, results):
    for rule in ['sum', 'mult']:
        cm = results[rule]['confusion_matrix']
        figsize = (25, 25)
        fig, axis = plt.subplots(figsize=figsize)
        filename = os.path.join(path, 'cm+%s.png' % rule)
        logging.info('[PNG] confusion matrix %s created' % filename)

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
        ytick_labels = [i[0] + ' (%s)' % i[1] for i in
                        zip(list_info_level['levels'].values(), list_info_level['count'].values())]

        axis.set_xticks(posix_xtick, labels=xtick_labels, fontsize=8, rotation=90)
        axis.set_yticks(posix_ytick, labels=ytick_labels, fontsize=8, rotation=0)
        axis.set_xlabel('True label', fontsize=14)
        axis.set_ylabel('Prediction label', fontsize=14)
        axis.set_facecolor('white')
        axis.set_title('test', fontsize=24, pad=32)

        plt.ioff()
        plt.tight_layout()
        plt.savefig(filename, format='png', bbox_inches='tight', dpi=300)
        plt.cla()
        plt.clf()
        plt.close(fig)


def figure_topk(filename, title, x, y):
    fig_size = (10, 10)
    figure, axis = plt.subplots(figsize=fig_size)
    plt.plot(x, y, marker='o', color='green')

    axis.set_xlabel('k', fontsize=20)
    axis.set_ylabel('Correct labels', fontsize=20)
    axis.set_facecolor('white')
    axis.set_title(title, fontsize=28, pad=20)

    # Be sure to only pick integer tick locations.
    for axis in [axis.xaxis, axis.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.grid(True)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.25)
    plt.ioff()
    plt.tight_layout()
    plt.savefig(filename, format='png', bbox_inches='tight', dpi=300)
    plt.cla()
    plt.clf()
    plt.close(figure)
