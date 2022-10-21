import csv
import os
import pathlib

import numpy as np
import pandas as pd
import sklearn.metrics
from matplotlib import pyplot as plt

ROUND_VALUE = 2


def save_fold(cfg, classifier_name, dataset, list_result_fold, list_time, path):
    list_files = []
    for fold in range(0, cfg['fold']):
        list_fold = list(filter(lambda x: x['fold'] == fold, list_result_fold))
        time_fold = list(filter(lambda x: x['fold'] == fold, list_time))

        path_fold = os.path.join(path, str(fold))
        pathlib.Path(path_fold).mkdir(parents=True, exist_ok=True)

        confusion_matrix_by_fold(classifier_name, dataset, list_fold, path_fold)

        index, values = get_values_by_fold_and_metric(list_fold, 'accuracy')
        list_files.append({'filename': 'accuracy', 'index': index, 'path': path_fold, 'values': values})

        index, values = get_values_by_fold_and_metric(list_fold, 'f1_score')
        list_files.append({'filename': 'f1', 'index': index, 'path': path_fold, 'values': values})

        get_top_k_by_rule(list_fold, path_fold)

        index, values = info_by_fold(list_fold, time_fold)
        list_files.append({'filename': 'info_by_fold', 'index': index, 'path': path_fold, 'values': values})

    return list_files


def info_by_fold(list_fold, time):
    index = ['best_rule_accuracy', 'best_accuracy',
             'best_rule_f1', 'best_f1',
             # 'best_rule_top_k', 'best_top_k',
             'time_train_valid', 'time_search_best_params']
    best_rule_accuracy = max(list_fold, key=lambda x: x['accuracy'])
    best_rule_f1 = max(list_fold, key=lambda x: x['f1_score'])

    time_train_valid = time[0]['time_train_valid']
    time_search_best_params = time[0]['time_search_best_params']

    values = [
        [best_rule_accuracy['rule']],
        [best_rule_accuracy['accuracy'], round(best_rule_accuracy['accuracy'] * 100, ROUND_VALUE)],
        [best_rule_f1['rule']], [best_rule_f1['f1_score'], round(best_rule_f1['f1_score'] * 100, ROUND_VALUE)],
        [time_train_valid, round(time_train_valid, ROUND_VALUE)],
        [time_search_best_params, round(time_search_best_params, ROUND_VALUE)]
    ]
    return index, values


def get_top_k_by_rule(list_fold, path_fold):
    for rule in ['max', 'prod', 'sum']:
        result = list(filter(lambda x: x['rule'] == rule, list_fold))
        if len(result) > 0:
            top_k = result[0]['top_k']
            fold = result[0]['fold']
            max_top_k = result[0]['max_top_k']
            min_top_k = result[0]['min_top_k']
            y_test = result[0]['y_true']
            index = ['rule', 'min_top_k', 'max_top_k', 'total']
            values = [rule, min_top_k, max_top_k, len(y_test)]
            df = pd.DataFrame(top_k)
            df_info = pd.DataFrame(values, index)

            path_top_k_rule = os.path.join(path_fold, 'top_k', rule)
            pathlib.Path(path_top_k_rule).mkdir(exist_ok=True, parents=True)
            save_plot_top_k(fold, max_top_k, min_top_k, path_top_k_rule, rule, top_k, y_test)

            df.to_csv(os.path.join(path_top_k_rule, f'top_k_{rule}.csv'), na_rep='', sep=';', index=False,
                  quoting=csv.QUOTE_ALL)
            df_info.to_csv(os.path.join(path_top_k_rule, f'info_top_k_{rule}.csv'), na_rep='', sep=';', header=False,
                  quoting=csv.QUOTE_ALL)

            path_top_k_rule_xlsx = os.path.join(path_top_k_rule, 'xlsx')
            pathlib.Path(path_top_k_rule_xlsx).mkdir(exist_ok=True, parents=True)
            df.to_excel(os.path.join(path_top_k_rule_xlsx, f'top_k_{rule}.xlsx'), na_rep='', engine='xlsxwriter',
                      index=False)
            df_info.to_excel(os.path.join(path_top_k_rule_xlsx, f'info_top_k_{rule}.xlsx'), na_rep='',
                           engine='xlsxwriter', header=False)


def save_plot_top_k(fold, max_top_k, min_top_k, path_fold, rule, top_k, y_test):
    for v in [3, 5]:
        a = list(filter(lambda x: x['k'] <= v, top_k))

        x = []
        y = []
        for k in a:
            x.append(k['k'])
            y.append(k['top_k_accuracy'])

        background_color = 'white'

        plt.plot(x, y, marker='o', color='green')
        plt.title(f'Top k accuracy, Rule: {rule}, k: {v}, Fold: {fold},\n Min. top k: {min_top_k}, Máx. top k: {max_top_k}, Número de testes: {len(y_test)}',
                  fontsize=14, pad=20)
        plt.xlabel('k', fontsize=14)
        plt.ylabel('Número de acertos', fontsize=14)
        plt.grid(True)
        plt.gcf().subplots_adjust(bottom=0.15, left=0.25)
        plt.rcParams['figure.facecolor'] = background_color
        plt.tight_layout()
        plt.savefig(os.path.join(path_fold, f'top_k_{rule}_k={v}.png'))
        plt.cla()
        plt.clf()
        plt.close()

    x = []
    y = []
    for k in top_k:
        x.append(k['k'])
        y.append(k['top_k_accuracy'])

    background_color = 'white'

    plt.plot(x, y, marker='o', color='green')
    plt.title(
        f'Top k accuracy, Rule: {rule}, k: Todos, Fold: {fold},\n Min. top k: {min_top_k}, Máx. top k: {max_top_k}, Número de testes: {len(y_test)}',
        fontsize=14, pad=20)
    plt.xlabel('k', fontsize=14)
    plt.ylabel('Número de acertos', fontsize=14)
    plt.grid(True)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.25)
    plt.rcParams['figure.facecolor'] = background_color
    plt.tight_layout()
    plt.savefig(os.path.join(path_fold, f'top_k_{rule}.png'))
    plt.cla()
    plt.clf()
    plt.close()


def get_values_by_fold_and_metric(list_fold, metric):
    index = []
    values = []
    for rule in ['max', 'prod', 'sum']:
        result = list(filter(lambda x: x['rule'] == rule, list_fold))
        if len(result) > 0:
            index.append(rule)
            value_metric = result[0][metric]
            round_value_metric = round(result[0][metric], ROUND_VALUE)
            values.append([value_metric, round_value_metric])
    return index, values


def confusion_matrix_by_fold(classifier_name, dataset, list_fold, path_fold):
    for rule in ['max', 'prod', 'sum']:
        result = list(filter(lambda x: x['rule'] == rule, list_fold))
        if len(result) > 0:
            save_confusion_matrix(classifier_name, dataset, path_fold, result[0])


def save_confusion_matrix(classifier_name, dataset, path, result):
    filename = f'confusion_matrix_{result["rule"]}.png'
    # cinco labels -> IWSSIP
    # labels = ['$\it{Manekia}$', '$\it{Ottonia}$', '$\it{Peperomia}$', '$\it{Piper}$', '$\it{Pothomorphe}$']

    # acima de cinco labels -> dataset George -> ok
    # labels = ['$\it{aduncum}', '$\it{alata}', '$\it{amalago}', '$\it{arboreum}', '$\it{arifolia}', '$\it{barbarana}', '$\it{blanda}', '$\it{caldasianum}', '$\it{caldense}', '$\it{catharinae}', '$\it{cernuum}', '$\it{circinnata}', '$\it{corcovadensis}', '$\it{crassinervium}', '$\it{dilatatum}', '$\it{diospyrifolium}', '$\it{emarginella}', '$\it{galioides}', '$\it{gaudichaudianum}', '$\it{glabella}', '$\it{glabratum}', '$\it{hatschbachii}', '$\it{hayneanum}', '$\it{hilariana}', '$\it{hispidula}', '$\it{hispidum}', '$\it{hydrocotyloides}', '$\it{lhotzkianum}', '$\it{macedoi}', '$\it{malacophyllum}', '$\it{martiana}', '$\it{mikanianum}', '$\it{miquelianum}', '$\it{mollicomum}', '$\it{mosenii}', '$\it{nitida}', '$\it{obtusa}', '$\it{pereirae}', '$\it{pereskiaefolia}', '$\it{pereskiifolia}', '$\it{pseudoestrellensis}', '$\it{regnellii}', '$\it{reitzii}', '$\it{rhombea}', '$\it{rotundifolia}', '$\it{rupestris}', '$\it{solmsianum}', '$\it{subretusa}', '$\it{tetraphylla}', '$\it{trineura}', '$\it{trineuroides}', '$\it{umbellatum}', '$\it{urocarpa}', '$\it{viminifolium}', '$\it{xylosteoides}']

    # acima de cinco dez -> dataset George -> ok
    # labels = ['$\it{aduncum}', '$\it{alata}', '$\it{amalago}', '$\it{arboreum}', '$\it{barbarana}', '$\it{blanda}', '$\it{caldense}', '$\it{catharinae}', '$\it{cernuum}', '$\it{corcovadensis}', '$\it{crassinervium}', '$\it{dilatatum}', '$\it{gaudichaudianum}', '$\it{glabella}', '$\it{glabratum}', '$\it{hispidula}', '$\it{hispidum}', '$\it{malacophyllum}', '$\it{martiana}', '$\it{mikanianum}', '$\it{miquelianum}', '$\it{mollicomum}', '$\it{nitida}', '$\it{pereskiaefolia}', '$\it{pseudoestrellensis}', '$\it{regnellii}', '$\it{reitzii}', '$\it{rotundifolia}', '$\it{solmsianum}', '$\it{tetraphylla}', '$\it{trineura}', '$\it{urocarpa}', '$\it{viminifolium}', '$\it{xylosteoides}']

    # acima de cinco vinte -> dataset George -> ok
    labels = ['$\it{aduncum}$', '$\it{amalago}$', '$\it{arboreum}$', '$\it{blanda}$', '$\it{caldense}$', '$\it{catharinae}$', '$\it{corcovadensis}$', '$\it{crassinervium}$', '$\it{gaudichaudianum}$', '$\it{glabella}$', '$\it{glabratum}$', '$\it{hispidum}$', '$\it{martiana}$', '$\it{mikanianum}$', '$\it{miquelianum}$', '$\it{rotundifolia}$', '$\it{solmsianum}$', '$\it{tetraphylla}$', '$\it{urocarpa}$', '$\it{xylosteoides}$']

    # todas as labels -> dataset George
    # labels = ['$\it{abutiloides}', '$\it{aduncum}', '$\it{aequale}', '$\it{alata}', '$\it{alnoides}', '$\it{amalago}', '$\it{amplum}', '$\it{arboreum}', '$\it{arifolia}', '$\it{balansana}', '$\it{barbarana}', '$\it{blanda}', '$\it{brasiliensis}', '$\it{caldasianum}', '$\it{caldense}', '$\it{callosum}', '$\it{calophylla}', '$\it{catharinae}', '$\it{caulibarbis}', '$\it{cernuum}', '$\it{circinnata}', '$\it{clivicola}', '$\it{concinnatoris}', '$\it{corcovadensis}', '$\it{crassinervium}', '$\it{crinicaulis}', '$\it{delicatula}', '$\it{diaphanodies}', '$\it{diaphanoides}', '$\it{dilatatum}', '$\it{diospyrifolium}', '$\it{elongata}', '$\it{emarginella}', '$\it{flavicans}', '$\it{fuligineum}', '$\it{galioides}', '$\it{gaudichaudianum}', '$\it{glabella}', '$\it{glabratum}', '$\it{glaziovi}', '$\it{glaziovii}', '$\it{gracilicaulis}', '$\it{hatschbachii}', '$\it{hayneanum}', '$\it{hemmandorfii}', '$\it{hemmendorffii}', '$\it{hemmendorfii}', '$\it{hernandiifolia}', '$\it{hilariana}', '$\it{hispidula}', '$\it{hispidum}', '$\it{hydrocotyloides}', '$\it{ibiramana}', '$\it{lanceolato-peltata}', '$\it{lanceolatopeltata}', '$\it{lepturum}', '$\it{leucaenum}', '$\it{leucanthum}', '$\it{lhotzkianum}', '$\it{lhotzkyanum}', '$\it{lindbergii}', '$\it{lucaeanum}', '$\it{lyman-smithii}', '$\it{macedoi}', '$\it{magnoliifolia}', '$\it{malacophyllum}', '$\it{mandiocana}', '$\it{mandioccana}', '$\it{martiana}', '$\it{michelianum}', '$\it{mikanianium}', '$\it{mikanianum}', '$\it{miquelianum}', '$\it{mollicomum}', '$\it{mosenii}', '$\it{}', '$\it{nitida}', '$\it{nudifolia}', '$\it{obtusa}', '$\it{obtusifolia}', '$\it{ouabianae}', '$\it{ovatum}', '$\it{pellucida}', '$\it{pereirae}', '$\it{pereskiaefolia}', '$\it{pereskiifolia}', '$\it{perlongicaulis}', '$\it{permucronatum}', '$\it{piritubanum}', '$\it{pseudoestrellensis}', '$\it{pseudolanceolatum}', '$\it{psilostachya}', '$\it{punicea}', '$\it{quadrifolia}', '$\it{radicosa}', '$\it{reflexa}', '$\it{regenelli}', '$\it{regnellii}', '$\it{reitzii}', '$\it{renifolia}', '$\it{retivenulosa}', '$\it{rhombea}', '$\it{rivinoides}', '$\it{rizzinii}', '$\it{rotundifolia}', '$\it{rubricaulis}', '$\it{rupestris}', '$\it{sandersii}', '$\it{schwackei}', '$\it{solmsianum}', '$\it{stroemfeltii}', '$\it{subcinereum}', '$\it{subemarginata}', '$\it{subretusa}', '$\it{subrubrispica}', '$\it{subternifolia}', '$\it{tenuissima}', '$\it{tetraphylla}', '$\it{trichocarpa}', '$\it{trineura}', '$\it{trineuroides}', '$\it{tuberculatum}', '$\it{umbellata}', '$\it{umbellatum}', '$\it{urocarpa}', '$\it{vicosanum}', '$\it{viminifolium}', '$\it{warmingii}', '$\it{xylosteoides}', '$\it{xylosteroides}']

    # duas labels -> dataset George
    # labels = ['$\it{Peperomia}$', '$\it{Piper}$']

    confusion_matrix = sklearn.metrics.ConfusionMatrixDisplay(result['confusion_matrix'], display_labels=labels)
    # confusion_matrix.plot(cmap='Reds')

    title = f'Confusion Matrix\ndataset: {dataset}, classifier: {classifier_name}\naccuracy: {round(result["accuracy"], ROUND_VALUE)}, rule: {result["rule"]}'
    fontsize_title = 18
    pad_title = 20

    fontsize_labels = 14

    background_color = 'white'
    # plt.ioff()
    # plt.title(title, fontsize=fontsize_title, pad=pad_title)

    # rotation = 45
    rotation = 90

    figure, axis = plt.subplots(figsize=(10, 10))

    confusion_matrix.plot(ax=axis, cmap='Reds')

    axis.set_title(title, fontsize=fontsize_title, pad=pad_title)
    axis.set_xlabel('y_true', fontsize=fontsize_labels)
    axis.set_ylabel('y_pred', fontsize=fontsize_labels)

    plt.ioff()
    plt.xticks(np.arange(len(labels)), rotation=rotation, fontsize=fontsize_labels)
    plt.yticks(np.arange(len(labels)), fontsize=fontsize_labels)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.25)
    plt.rcParams['figure.facecolor'] = background_color
    plt.tight_layout()

    path_confusion_matrix=os.path.join(path, 'confusion_matrix')
    pathlib.Path(path_confusion_matrix).mkdir(exist_ok=True, parents=True)

    plt.savefig(os.path.join(path_confusion_matrix, filename), bbox_inches='tight', dpi=300)
    plt.cla()
    plt.clf()
    plt.close()
