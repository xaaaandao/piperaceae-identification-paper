import datetime
import os

import sklearn.ensemble
import sklearn.model_selection
import sklearn.neighbors
import sklearn.neural_network
import sklearn.preprocessing
import sklearn.svm
import sklearn.tree

from handcraft import handcraft
from non_handcraft import non_handcraft


def main():
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
    kf = sklearn.model_selection.KFold(n_splits=cfg['fold'], shuffle=True, random_state=cfg['seed'], )
    list_data_input = [
        '../dataset_gimp/imagens_george/features/RGB/segmented_unet/256/mobilenetv2/horizontal/patch=3/deep_feature/genus',
        '../dataset_gimp/imagens_george/features/RGB/segmented_unet/256/resnet50v2/horizontal/patch=3/deep_feature/genus',
        '../dataset_gimp/imagens_george/features/RGB/segmented_unet/256/vgg16/horizontal/patch=3/deep_feature/genus',
    ]
    for data in list_data_input:
        if len(os.listdir(data)) == 0:
            raise ValueError(f'has not data input {data}')

    handcraft(cfg, current_datetime, kf, list_data_input, list_extractor)

    non_handcraft(cfg, current_datetime, kf, list_data_input, list_extractor)


if __name__ == '__main__':
    main()
