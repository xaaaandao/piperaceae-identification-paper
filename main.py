import os

import click
import datetime

from handcraft import handcraft
from non_handcraft import non_handcraft


@click.command()
@click.option('-i', '--input', multiple=True)
@click.option('-l', '--filename_labels')
def main(input, filename_labels):
    cfg = {
        'fold': 5,
        'n_jobs': -1,
        'seed': 1234,
        'dir_input': '../dataset/features',
        'dir_output': 'out'
    }

    list_extractor = {
        'lbp': [59],
        'surf64': [128, 256, 257],
        'surf128': [128, 256, 513],
        'mobilenetv2': [128, 256, 512, 1024, 1280],
        'resnet50v2': [128, 256, 512, 1024, 2048],
        'vgg16': [128, 256, 512]
    }

    current_datetime = datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
    list_data_input = [
    ]
    if len(input) == 0 and len(list_data_input) == 0:
        raise ValueError(f'list data input is empty')

    list_data_input = list_data_input + [i for i in list(input) if i not in list_data_input]

    print(f'quantidade {len(list_data_input)}')
    handcraft(cfg, current_datetime, filename_labels, list_data_input, list_extractor)
    non_handcraft(cfg, current_datetime, filename_labels, list_data_input, list_extractor)


if __name__ == '__main__':
    main()
