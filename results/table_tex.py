import os
import pathlib

import numpy as np
import pandas as pd

path = '/media/xandao/c2f58d30-ff2c-47f7-95af-91ad6fd69760/resultados/'
list_files_xlsx = [p for p in pathlib.Path(path).rglob('*.xlsx') if p.is_file()]


def model_table(color_mode, df, image_size, metric):
    return "\\begin{table}[H]\n" + \
            "    \\caption{?}\n" + \
            "    \\label{%s:%s:%s}\n" % (color_mode, image_size, metric) + \
            "    \\centering\n" + \
            "    \\begin{tabular}{llll}\n" + \
            "        \\hline\n" + \
            header_table(color_mode) + \
            lines('DT', 'DecisionTreeClassifier', color_mode, df, image_size, metric, 'unet') + \
            lines('k-NN', 'KNeighborsClassifier', color_mode, df, image_size, metric, 'unet') + \
            lines('MLP', 'MLPClassifier', color_mode, df, image_size, metric, 'unet') + \
            lines('RF', 'RandomForestClassifier', color_mode, df, image_size, metric, 'unet') + \
            lines('SVM', 'SVC', color_mode, df, image_size, metric, 'unet') + \
            "        \\hline \n" + \
            "    \\end{tabular} \n" + \
            "\\end{table}".replace('±', '')


def header_grayscale():
    return "        \\multirow{3}{*}{\\textbf{Classifier}} & \multicolumn{5}{c}{\\textbf{Descriptors}} \\\\ \n" + \
        "        & \\multicolumn{1}{c}{\\textbf{LBP}} & \\multicolumn{1}{c}{\\textbf{SURF}} & \\multicolumn{1}{c}{\\textbf{MobileNet-V2}} & \\multicolumn{1}{c}{\\textbf{ResNet50}} & \\multicolumn{1}{c}{\\textbf{VGG16}} \\\\ \\hline \n"


def header_rgb():
    return "        \\multirow{3}{*}{\\textbf{Classifier}} & \multicolumn{3}{c}{\\textbf{Descriptors}} \\\\ \n" + \
        "        & \\multicolumn{1}{c}{\\textbf{MobileNet-V2}} & \\multicolumn{1}{c}{\\textbf{ResNet50}} & \\multicolumn{1}{c}{\\textbf{VGG16}} \\\\ \\hline \n"


def header_table(color_mode):
    return header_rgb() if 'rgb' in color_mode else header_grayscale()


def lines_rgb(classifier, classifier_fullname, df, image_size, metric, segmented):
    return \
            "        %s & \\accstd{%s}{%s} & \\accstd{%s}{%s} & \\accstd{%s}{%s} \\\\ \n" % \
            (classifier,
             get_mean(classifier_fullname, df, 'mobilenetv2_1280', image_size, metric, segmented),
             get_std(classifier_fullname, df, 'mobilenetv2_1280', image_size, metric, segmented),
             get_mean(classifier_fullname, df, 'resnet50v2_2048', image_size, metric, segmented),
             get_std(classifier_fullname, df, 'resnet50v2_2048', image_size, metric, segmented),
             get_mean(classifier_fullname, df, 'vgg16_512', image_size, metric, segmented),
             get_std(classifier_fullname, df, 'vgg16_512', image_size, metric, segmented))


def get_mean(classifier_fullname, df, extractor, image_size, metric, segmented):
    if 'top_k' in metric:
        values = df.loc[f'{extractor}_{metric}', f'{classifier_fullname}_{image_size}_{segmented}']
        if not 'nan' in str(values):
            return values.split('-')[0]
        return 'nan'
    return df.loc[f'{extractor}_{metric}', f'{classifier_fullname}_{image_size}_{segmented}']


def get_std(classifier_fullname, df, extractor, image_size, metric, segmented):
    if 'top_k' in metric:
        values = df.loc[f'{extractor}_{metric}', f'{classifier_fullname}_{image_size}_{segmented}']
        if not 'nan' in str(values):
            return values.split('-')[1]
        return 'nan'
    return df.loc[f'{extractor}_std', f'{classifier_fullname}_{image_size}_{segmented}']


def lines_grayscale(classifier, classifier_fullname, df, image_size, metric, segmented):
    return \
            "        %s & \\accstd{%s}{%s} & \\accstd{%s}{%s} & \\accstd{%s}{%s} & \\accstd{%s}{%s} & \\accstd{%s}{%s} \\\\ \n" % \
            (classifier,
             get_mean(classifier_fullname, df, 'lbp_59', image_size, metric, segmented),
             get_std(classifier_fullname, df, 'lbp_59', image_size, metric, segmented),
             get_mean(classifier_fullname, df, 'surf64_257', image_size, metric, segmented),
             get_std(classifier_fullname, df, 'surf64_257', image_size, metric, segmented),
             get_mean(classifier_fullname, df, 'mobilenetv2_1280', image_size, metric, segmented),
             get_std(classifier_fullname, df, 'mobilenetv2_1280', image_size, metric, segmented),
             get_mean(classifier_fullname, df, 'resnet50v2_2048', image_size, metric, segmented),
             get_std(classifier_fullname, df, 'resnet50v2_2048', image_size, metric, segmented),
             get_mean(classifier_fullname, df, 'vgg16_512', image_size, metric, segmented),
             get_std(classifier_fullname, df, 'vgg16_512', image_size, metric, segmented))

             # df.loc[f'lbp_59_{metric}', f'{classifier_fullname}_{image_size}_{segmented}'],
             # df.loc[f'lbp_59_std', f'{classifier_fullname}_{image_size}_{segmented}'],
             # df.loc[f'_{metric}', f'{classifier_fullname}_{image_size}_{segmented}'],
             # df.loc[f'surf64_257_std', f'{classifier_fullname}_{image_size}_{segmented}'],
             # df.loc[f'mobilenetv2_1280_{metric}', f'{classifier_fullname}_{image_size}_{segmented}'],
             # df.loc[f'mobilenetv2_1280_std', f'{classifier_fullname}_{image_size}_{segmented}'],
             # df.loc[f'resnet50v2_2048_{metric}', f'{classifier_fullname}_{image_size}_{segmented}'],
             # df.loc[f'resnet50v2_2048_std', f'{classifier_fullname}_{image_size}_{segmented}'],
             # df.loc[f'vgg16_512_{metric}', f'{classifier_fullname}_{image_size}_{segmented}'],
             # df.loc[f'vgg16_512_std', f'{classifier_fullname}_{image_size}_{segmented}'])


def lines(classifier, classifier_fullname, color_mode, df, image_size, metric, segmented):
    return lines_rgb(classifier, classifier_fullname, df, image_size, metric, segmented) if 'rgb' in color_mode else lines_grayscale(classifier, classifier_fullname, df, image_size, metric, segmented)


list_classifiers = ['DecisionTreeClassifier', 'KNeighborsClassifier', 'MLPClassifier', 'RandomForestClassifier', 'SVC']


for color_mode in ['rgb', 'grayscale']:
    list_filename = [xlsx for xlsx in list_files_xlsx if 'mean_' + str(color_mode) in str(xlsx.resolve())]
    for file in list_filename:
        print(file)
        df = pd.read_excel(str(file.resolve()), header=0, index_col=0)
        for metric in ['mean', 'top_k']:
            all = ""
            folder = os.path.join(str(file.parents[0]), 'tex')
            pathlib.Path(folder).mkdir(exist_ok=True, parents=True)
            for image_size in ['256', '400', '512']:
                # filename_out = os.path.join(str(file.parents[0]), folder, '%s_%s.tex' % (image_size, metric))
                out = model_table(color_mode, df, image_size, metric)
                all = all + out + "\n\n"
                # with open(filename_out, 'w') as f:
                #     f.write(out)
                #     f.close()
            filename_out = os.path.join(str(file.parents[0]), folder, '%s.tex' % (metric))
            with open(filename_out, 'w') as f:
                f.write(all)
                f.close()