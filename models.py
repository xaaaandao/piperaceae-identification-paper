import dataclasses

class Model:
    models: dict = {
        'mobilenetv2': [1280, 1024, 512, 256, 128],
        'vgg16': [512, 256, 128],
        'resnet50v2': [2048, 1024, 512, 256, 128],
        'lbp': [59],
        'surf64': [257, 256, 128],
        'surf128': [513, 512, 256, 128]
    }
