import os

import click
import datetime

from handcraft import handcraft
from non_handcraft import non_handcraft


@click.command()
@click.option('-i', '--input', multiple=True)
def main(input):
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
        # '/home/xandao/Documentos/GitHub/dataset_gimp/imagens_george/features/RGB/segmented_unet/256/mobilenetv2/patch=3/horizontal/genus/peperomia-piper',
        # '../dataset_gimp/imagens_george/features/grayscale/segmented_unet/256/resnet50v2/horizontal/patch=3/deep_feature/specific_epithet/acima-20',
        # '../dataset_gimp/imagens_george/features/grayscale/segmented_unet/256/vgg16/horizontal/patch=3/deep_feature/specific_epithet/acima-20',
        # '../dataset_gimp/imagens_george/features/grayscale/segmented_unet/400/mobilenetv2/horizontal/patch=3/deep_feature/specific_epithet/acima-20',
        # '../dataset_gimp/imagens_george/features/grayscale/segmented_unet/400/resnet50v2/horizontal/patch=3/deep_feature/specific_epithet/acima-20',
        # '../dataset_gimp/imagens_george/features/grayscale/segmented_unet/400/vgg16/horizontal/patch=3/deep_feature/specific_epithet/acima-20',
        # '../dataset_gimp/imagens_george/features/grayscale/segmented_unet/512/mobilenetv2/horizontal/patch=3/deep_feature/specific_epithet/acima-20',
        # '../dataset_gimp/imagens_george/features/grayscale/segmented_unet/512/resnet50v2/horizontal/patch=3/deep_feature/specific_epithet/acima-20',
        # '../dataset_gimp/imagens_george/features/grayscale/segmented_unet/512/vgg16/horizontal/patch=3/deep_feature/specific_epithet/acima-20',

    ]
    if len(input) == 0 and len(list_data_input) == 0:
        raise ValueError(f'list data input is empty')

    list_data_input = list_data_input + input

    for data in list_data_input:
        if len(os.listdir(data)) == 0:
            raise ValueError(f'has not data input {data}')

    handcraft(cfg, current_datetime, list_data_input, list_extractor)

    non_handcraft(cfg, current_datetime, list_data_input, list_extractor)


if __name__ == '__main__':
    main()
