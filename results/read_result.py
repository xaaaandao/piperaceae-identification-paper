
import csv
import shutil

import click
import datetime

import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
import pathlib

ROUND_VALUE = 3


def get_index_time(list_extractor):
    return [e + '_' + str(d) + '_' + m for e in list_extractor.keys() for d in reversed(list_extractor[e]) for m in
            ['train_test', 'gridsearch']]


def get_index_folder(list_extractor):
    return [e + '_' + str(d) + '_' + m for e in list_extractor.keys() for d in reversed(list_extractor[e]) for m in
            ['folder']]


def create_df(list_classifier, list_extractor, list_dim, list_segmented):
    return {
        'mean': pd.DataFrame(index=get_index_mean(list_extractor), columns=get_columns(list_classifier, list_dim, list_segmented)),
        'time': pd.DataFrame(index=get_index_time(list_extractor), columns=get_columns(list_classifier, list_dim, list_segmented)),
        'folder': pd.DataFrame(index=get_index_folder(list_extractor), columns=get_columns(list_classifier, list_dim, list_segmented))
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
        return str(percentage).replace('.', ',')
    return str(0)
    # return str(top_k)


def insert_sheet(column, date, df, index_folder, index_gridsearch, index_mean, index_std, index_top_k, index_train_test, mean, mean_time_search_best_params, mean_time_train_valid, std, top_k, total_top_k):
    df['mean'].loc[index_mean, column] = round_mean(mean)
    df['mean'].loc[index_std, column] = plus_minus_std(std)
    df['mean'].loc[index_top_k, column] = get_top_k(top_k, total_top_k)
    df['time'].loc[index_train_test, column] = round_time(mean_time_train_valid)
    df['time'].loc[index_gridsearch, column] = round_time(mean_time_search_best_params)
    df['folder'].loc[index_folder, column] = date


def get_csv(filename, header=None):
    return pd.read_csv(filename, sep=';', index_col=0, header=header)


def fill_sheet_mean_std(classifier, date, df, filename, image_size, extractor, n_features, n_patch, plot, segmented):
    sheet_mean = get_csv(filename)
    filename_mean = 'mean.csv'
    metric = 'f1'
    mean = sheet_mean.loc['mean_%s_sum' % metric][1]
    mean_time_search_best_params = sheet_mean.loc['mean_time_search_best_params'][1]
    mean_time_train_valid = sheet_mean.loc['mean_time_train_valid'][1]
    std = sheet_mean.loc['std_%s_sum' % metric][1]

    filename_mean_top_k_sum = str(filename).replace(filename_mean, 'mean_top_k/mean_top_sum.csv')
    if os.path.exists(filename_mean_top_k_sum):
        sheet_mean_top_k_sum = get_csv(filename_mean_top_k_sum, header=0)
        top_k = sheet_mean_top_k_sum.iloc[0]['top_k']
    else:
        top_k = 0

    filename_info_top_k_sum = str(filename).replace(filename_mean, '0/top_k/sum/info_top_k_sum.csv')
    if os.path.exists(filename_info_top_k_sum):
        sheet_info_top_k_sum = get_csv(filename_info_top_k_sum)
        total_top_k = sheet_info_top_k_sum.loc['total'][1]
    else:
        total_top_k = 0
    # total_top_k = 1
    print(top_k)
    print(total_top_k)

    index_mean = extractor + '_' + n_features + '_' + 'mean'
    index_std = extractor + '_' + n_features + '_' + 'std'
    index_top_k = extractor + '_' + n_features + '_' + 'top_k'
    index_train_test = extractor + '_' + n_features + '_' + 'train_test'
    index_gridsearch = extractor + '_' + n_features + '_' + 'gridsearch'
    index_folder = extractor + '_' + n_features + '_' + 'folder'
    column = classifier + '_' + image_size + '_' + segmented

    plot.append({
        'extractor': extractor,
        'n_features': n_features,
        'classifier': classifier,
        'image_size': image_size,
        'mean': mean
    })


    insert_sheet(column, date, df, index_folder, index_gridsearch, index_mean, index_std, index_top_k, index_train_test, mean, mean_time_search_best_params, mean_time_train_valid, std, top_k, total_top_k)


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


def get_threshold(dir_input, taxon):
    if '%s/5' % taxon in dir_input:
        return 5
    elif '%s/10' % taxon in dir_input:
        return 10
    elif '%s/20' % taxon in dir_input:
        return 20
    return 0


@click.command()
@click.option(
    '--color',
    '-c',
    type=click.Choice(['RGB', 'grayscale', 'rgb']),
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
@click.option(
    '--taxon',
    type=click.Choice(['specific_epithet', 'genus']),
    required=True
)
@click.option(
    '--threshold',
    type=int,
    required=True
)
def main(color, input, taxon, threshold, output):
    if not os.path.exists(input):
        raise NotADirectoryError('directory %s not exists' % input)

    if not os.path.exists(output):
        raise NotADirectoryError('directory %s not exists' % output)

    list_files = [file for file in pathlib.Path(input).rglob('mean.csv') if file.is_file() and color.lower() in str(file.resolve())]

    if len(list_files) == 0:
        raise ValueError('list empty')

    list_extractor = {
        'lbp': [59],
        'surf64': [128, 256, 257],
        'surf128': [128, 256, 513],
        'mobilenetv2': [128, 256, 512, 1024, 1280],
        'resnet50v2': [128, 256, 512, 1024, 2048],
        'vgg16': [128, 256, 512]
    }

    list_classifier = ['DecisionTreeClassifier', 'KNeighborsClassifier', 'MLPClassifier', 'RandomForestClassifier',
                       'SVC']
    list_dim = [256, 400, 512]

    list_segmented = ['unet']
    # list_segmented = ['unet', 'manual']

    df = create_df(list_classifier, list_extractor, list_dim, list_segmented)


    import tarfile
    for filename in pathlib.Path(input).rglob('top_k.tar.gz'):
        try:
            t = tarfile.open(filename.absolute(), 'r')
        except IOError as e:
            print(e)
        else:
            t.extractall(members=[m for m in t.getmembers() if 'info_top_k_sum.csv' in m.name])


    path_zip = './out'
    for folder in pathlib.Path(path_zip).glob('*'):
        os.system('rsync -av %s %s' % (folder, input))

    shutil.rmtree(path_zip)

    plot = []
    for file in sorted(list_files):
        sheet_info = pd.read_csv(str(file).replace('mean.csv', 'info.csv'), header=None, sep=';', index_col=0)

        classifier = get_classifier(list_classifier, file)
        color = sheet_info.loc['color_mode'][1]
        image_size = sheet_info.loc['dim_image'][1]
        extractor = sheet_info.loc['extractor'][1]
        file_threshold = get_threshold(sheet_info.loc['dir_input'][1], taxon)
        n_features = sheet_info.loc['data_n_features'][1]
        n_patch = sheet_info.loc['n_patch'][1]
        slice_patch = sheet_info.loc['slice'][1]
        segmented = get_type_segmented(file)
        date = get_date(file)

        print(file_threshold, classifier, image_size, color, extractor, n_features, n_patch, slice_patch, segmented, date)

        if threshold == file_threshold:
            fill_sheet_mean_std(classifier, date, df, file, image_size, extractor, n_features, n_patch, plot, segmented)

    save_df(color, df, output)


def get_columns(list_classifier, list_dim, list_segmented):
    columns = [c + '_' + str(d) + '_' + s for c in list_classifier for s in sorted(list_segmented) for d in list_dim]
    return columns


def get_index_mean(list_extractor):
    return [e + '_' + str(d) + '_' + m for e in list_extractor.keys() for d in reversed(list_extractor[e]) for m in
             ['mean', 'std', 'top_k']]


def plot_mean(output, plot):
    image_size = '256'
    bar_width = 0.25
    for n_features in [128, 256, 512, 1024, 1280, 2048]:
        mean_mobilenet = get_list_mean('mobilenetv2', image_size, n_features, plot)
        mean_vgg = get_list_mean('vgg', image_size, n_features, plot)
        mean_resnet = get_list_mean('resnet', image_size, n_features, plot)

        labels_axis_x = ['DecisionTree', 'k-NN', 'MLP', 'RF', 'SVM']
        x_axis = np.arange(len(labels_axis_x))

        figure, axis = plt.subplots(figsize=(12, 6))

        add_mean_plot(axis, bar_width, 'mobilenetv2', mean_mobilenet, x_axis)
        add_mean_plot(axis, bar_width, 'resnet50v2', mean_resnet, x_axis + 0.25)
        add_mean_plot(axis, bar_width, 'vgg16', mean_vgg, x_axis + 0.5)
        add_bar_label(axis)

        r = np.arange(len(labels_axis_x))
        plt.xticks(r + 0.4 / 2, labels_axis_x)
        plt.title('Mean F1-score (n_features: %s)' % n_features, fontweight='bold', fontsize='xx-large', pad=20)
        plt.xlabel('classifiers', fontweight='bold', fontsize='xx-large')
        plt.ylabel('mean', fontweight='bold', fontsize='xx-large')
        pathlib.Path(os.path.join(output, 'plots')).mkdir(exist_ok=True, parents=True)
        filename = os.path.join(output, 'plots', 'f1_%s.png' % n_features)
        print('[TOP-k] save %s' % filename)
        plt.savefig(filename, dpi=300)
        plt.grid()
        plt.cla()
        plt.clf()
        plt.close(figure)


def add_mean_plot(ax, bar_width, label, mean, x_axis):
    if len(mean) > 0:
        ax.bar(x_axis, mean, width=bar_width, edgecolor='black', label=label)
        ax.legend(fontsize='x-large')


def save_df(color, df, dir_output):
    filename_mean = os.path.join(dir_output, 'mean_%s' % color)
    filename_mean_time = os.path.join(dir_output, 'mean_time_%s' % color)
    filename_date = os.path.join(dir_output, 'date_%s' % color)
    save_xlsx(df['mean'], filename_mean)
    save_xlsx(df['time'], filename_mean_time)
    save_xlsx(df['folder'], filename_date)


def save_xlsx(df, filename):
    df.to_excel('%s.xlsx' % filename, na_rep='', engine='xlsxwriter')


if __name__ == '__main__':
    main()





