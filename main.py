import click
import datetime

from save.save_confusion_matrix import get_list_label
from test import check_input

cfg = {
    'fold': 5,
    'n_jobs': -1,
    'seed': 1234,
    'only_find_best_model': False,
    'dir_input': '../dataset/features',
    'dir_output': './out',
    'verbose': 42
}

list_extractor = {
    'lbp': [59],
    'surf64': [128, 256, 257],
    'surf128': [128, 256, 513],
    'mobilenetv2': [128, 256, 512, 1024, 1280],
    'resnet50v2': [128, 256, 512, 1024, 2048],
    'vgg16': [128, 256, 512]
}


def check_if_has_input(list_data_input, list_user_input):
    return list_input_is_empty(list_user_input) and list_input_is_empty(list_data_input)


def list_input_is_empty(list_input):
    return len(list_input) == 0


@click.command()
@click.option('-i', '--list_user_input', multiple=True, default=['/home/xandao/Documentos/resultados_gimp/identificacao_george/especie/20'])
@click.option('-l', '--labels', default=['/home/xandao/Documentos/GitHub/dataset_gimp/imagens_george/imagens/RGB/specific_epithet/256/20/label2.txt'])
@click.option('-m', '--metric', type=click.Choice(['f1_weighted', 'accuracy']), default='f1_weighted')
def main(list_user_input, labels, metric):
    current_datetime = datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
    list_data_input = []

    if check_if_has_input(list_data_input, list_user_input):
        raise ValueError('list data input is empty')

    list_data_input = list_data_input + [i for i in list(list_user_input) if i not in list_data_input]
    print('[INFO] quantidade de entradas: %d, filename labels: %s' % (len(list_data_input), str(labels)))
    list_labels = get_list_label(labels)

    if check_if_has_data_and_list_labels(list_data_input, list_labels):
        check_input(cfg, current_datetime, list_labels, list_data_input, list_extractor, metric)


def check_if_has_data_and_list_labels(list_data_input, list_labels):
    return data_is_not_empty(list_data_input) and list_labels_is_not_empty(list_labels)


def data_is_not_empty(list_data_input):
    return len(list_data_input) > 0


def list_labels_is_not_empty(list_labels):
    return len(list_labels) > 0


if __name__ == '__main__':
    main()
