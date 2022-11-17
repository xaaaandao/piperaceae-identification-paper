import csv
import click
import datetime

import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
import pathlib

ROUND_VALUE = 3


def create_df(columns, index):
    return {
        'mean': pd.DataFrame(index=index, columns=columns),
        'time': pd.DataFrame(index=index, columns=columns),
        'folder': pd.DataFrame(index=index, columns=columns)
    }


def round_mean(value):
    return str(round(value * 100, ROUND_VALUE)).replace('.', ',')


def round_time(value):
    return str(round(value, ROUND_VALUE))


def plus_minus_std(value):
    return "±" + str(round(value, ROUND_VALUE))


def get_classifier(list_classifier, path):
    classifier = list(filter(lambda x: x.lower() in str(path).lower(), list_classifier))

    if len(classifier) == 0:
        raise ValueError('classifier not available in list')

    return classifier[0]


def is_date(string):
    try:
        datetime.datetime.strptime(string, '%d-%m-%Y-%H-%M-%S')
        return True
    except ValueError:
        return False


def get_date(path):
    for p in str(path).split('/'):
        if is_date(p):
            return p
    return None


def get_top_k(top_k, total_top_k):
    if int(top_k) > 0 and int(total_top_k) > 0:
        percentage = int(top_k) / int(total_top_k)
        percentage = str(round(percentage * 100, 1))
        # return str(str(top_k) + '/' + str(total_top_k) + '=' + str(percentage) + '%')
        return str(percentage).replace('.', ',')
    return str(0)


def insert_sheet(column, date, df, index_mean, index_std, index_top_k, mean, mean_time_search_best_params, mean_time_train_valid, std, top_k, total_top_k):
    df['mean'].loc[index_mean, column] = round_mean(mean)
    df['mean'].loc[index_std, column] = plus_minus_std(std)
    df['mean'].loc[index_top_k, column] = get_top_k(top_k, total_top_k)
    df['time'].loc[index_mean, column] = round_time(mean_time_train_valid)
    df['time'].loc[index_std, column] = round_time(mean_time_search_best_params)
    # df['folder'].loc[index_folder, column] = date


def get_csv(filename, header=None):
    return pd.read_csv(filename, sep=';', index_col=0, header=header)


def fill_sheet_mean_std(classifier, date, df, filename, image_size, extractor, n_features, n_patch, plot, segmented):
    sheet_mean = get_csv(filename)
    mean = sheet_mean.loc['mean_f1_sum'][1]
    mean_time_search_best_params = sheet_mean.loc['mean_time_search_best_params'][1]
    mean_time_train_valid = sheet_mean.loc['mean_time_train_valid'][1]
    std = sheet_mean.loc['std_f1_sum'][1]

    # if not os.path.exists(str(filename).replace('mean.csv', 'mean_top_k/mean_top_k_sum.csv')):
    #     raise FileNotFoundError(f'file not exists {str(filename).replace("mean.csv", "mean_top_k/mean_top_k_sum.csv")}')
    #
    # if not os.path.exists(str(filename).replace('mean.csv', '0/top_k/sum/info_top_k_sum.csv')):
    #     raise FileNotFoundError(f'file not exists {str(filename).replace("mean.csv", "0/top_k/sum/info_top_k_sum.csv")}')
    #
    # sheet_mean_top_k_sum = get_csv(str(filename).replace('mean.csv', 'mean_top_k/mean_top_k_sum.csv'), header=0)
    # sheet_info_top_k_sum = get_csv(str(filename).replace('mean.csv', '0/top_k/sum/info_top_k_sum.csv'))
    # top_k = sheet_mean_top_k_sum.iloc[1]['top_k']
    # total_top_k = sheet_info_top_k_sum.loc['total'][1]
    top_k = 0
    total_top_k = 0

    index_mean = extractor + '_' + n_features + '_' + 'mean'
    index_std = extractor + '_' + n_features + '_' + 'std'
    index_top_k = extractor + '_' + n_features + '_' + 'top_k'
    column = classifier + '_' + image_size + '_' + segmented

    plot.append({
        'extractor': extractor,
        'n_features': n_features,
        'classifier': classifier,
        'image_size': image_size,
        'mean': mean
    })

    insert_sheet(column, date, df, index_mean, index_std, index_top_k, mean, mean_time_search_best_params, mean_time_train_valid, std, top_k, total_top_k)


def get_list_mean(extractor, image_size, n_features, plot):
    return [float(p['mean']) for p in plot if extractor in p['extractor'] and image_size in p['image_size'] and str(n_features) == p['n_features']]


def add_bar_label(axis):
    for container in axis.containers:
        axis.bar_label(container, label_type='center', rotation=90, color='white', fontsize='xx-large')


def get_type_segmented(file):
    return 'unet' if unet_not_hifen(file) or unet_hifen(file) else 'manual'


def unet_hifen(file):
    return 'u-net' in str(file).lower()


def unet_not_hifen(file):
    return 'unet' in str(file).lower()


@click.command()
@click.option(
    '--color',
    '-c',
    type=click.Choice(['RGB', 'grayscale']),
    required=True
)
@click.option(
    '--input',
    '-i',
    type=str,
    required=True
)
@click.option(
    '--output',
    '-o',
    type=str,
    required=True
)
def main(color, input, output):
    if not os.path.exists(input):
        raise NotADirectoryError(f'directory {input} not exists')

    if not os.path.exists(output):
        raise NotADirectoryError(f'directory {output} not exists')

    list_files = [file for file in pathlib.Path(input).rglob('mean.csv') if file.is_file()]
    if len(list_files) == 0:
        raise FileNotFoundError(f'files not found in directory {input}')

    list_extractor = {
        'lbp': [59],
        'surf64': [128, 256, 257],
        'surf128': [128, 256, 513],
        'mobilenetv2': [128, 256, 512, 1024, 1280],
        'resnet50v2': [128, 256, 512, 1024, 2048],
        'vgg16': [128, 256, 512]
    }
    index = [e + '_' + str(d) + '_' + m for e in list_extractor.keys() for d in reversed(list_extractor[e]) for m in
             ['mean', 'std', 'top_k']]

    list_classifier = ['DecisionTreeClassifier', 'KNeighborsClassifier', 'MLPClassifier', 'RandomForestClassifier',
                       'SVC']
    list_dim = [256, 400, 512]

    list_segmented = ['unet']
    columns = [c + '_' + str(d) + '_' + s for c in list_classifier for s in sorted(list_segmented) for d in list_dim]

    df = create_df(columns, index)

    plot = []
    for file in sorted(list_files):
        sheet_info = pd.read_csv(str(file).replace('mean.csv', 'info.csv'), header=None, sep=';', index_col=0)

        classifier = get_classifier(list_classifier, file)
        # color = sheet_info.loc['color_mode'][1]
        image_size = sheet_info.loc['dim_image'][1]
        extractor = sheet_info.loc['extractor'][1]
        n_features = sheet_info.loc['data_n_features'][1]
        n_patch = sheet_info.loc['n_patch'][1]
        slice_patch = sheet_info.loc['slice'][1]
        segmented = get_type_segmented(file)
        date = get_date(file)

        fill_sheet_mean_std(classifier, date, df, file, image_size, extractor, n_features, n_patch, plot, segmented)

    save_df(color, df, output)

    image_size = '256'
    bar_width = 0.25
    for n_features in [128, 256, 512, 1024, 1280, 2048]:
        mean_mobilenet = get_list_mean('mobilenetv2', image_size, n_features, plot)
        mean_vgg = get_list_mean('vgg', image_size, n_features, plot)
        mean_resnet = get_list_mean('resnet', image_size, n_features, plot)

        labels_axis_x = ['DecisionTree', 'k-NN', 'MLP', 'RF', 'SVM']
        X_axis = np.arange(len(labels_axis_x))

        figure, axis = plt.subplots(figsize=(12, 6))

        add_mean_plot(X_axis, axis, bar_width, 'mobilenetv2', mean_mobilenet)
        add_mean_plot(X_axis+0.25, axis, bar_width, 'resenet50v2', mean_resnet)
        add_mean_plot(X_axis+0.5, axis, bar_width, 'vgg16', mean_vgg)
        add_bar_label(axis)

        r = np.arange(len(labels_axis_x))
        plt.xticks(r + 0.4/2, labels_axis_x)
        plt.title(f'Mean F1-score (n_features: {n_features})', fontweight='bold', fontsize='xx-large', pad=20)
        plt.xlabel('classifiers', fontweight='bold', fontsize='xx-large')
        plt.ylabel('mean', fontweight='bold', fontsize='xx-large')
        pathlib.Path(os.path.join(output, 'plots')).mkdir(exist_ok=True, parents=True)
        filename = os.path.join(output, 'plots', f'f1_{n_features}.png')
        print(f'[TOP-k]save {filename}')
        plt.savefig(filename, dpi=300)
        plt.grid()
        plt.cla()
        plt.clf()
        plt.close(figure)


def add_mean_plot(X_axis, ax, bar_width, label, mean):
    if len(mean) > 0:
        ax.bar(X_axis, mean, width=bar_width, edgecolor='black', label=label)
        ax.legend(fontsize='x-large')


def save_df(color, df, dir_output):
    filename_mean = os.path.join(dir_output, f'mean_{color}')
    filename_mean_time = os.path.join(dir_output, f'mean_time_{color}')
    df['mean'].to_csv(f'{filename_mean}.csv', sep=';', na_rep='', quoting=csv.QUOTE_ALL)
    df['mean'].to_excel(f'{filename_mean}.xlsx', na_rep='', engine='xlsxwriter')
    df['time'].to_csv(f'{filename_mean_time}.csv', sep=';', na_rep='', quoting=csv.QUOTE_ALL)
    # df['time'].to_excel(f'{filename_mean_time}.xlsx', na_rep='', engine='xlsxwriter')


if __name__ == '__main__':
    main()





