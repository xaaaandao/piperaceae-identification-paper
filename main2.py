import os

from PIL import Image, ImageEnhance
import pathlib
import tensorflow as tf

for s in [5, 10, 20]:
    for p in pathlib.Path(f'/home/xandao/Documentos/pr_dataset+{s}').rglob('*/original/*/*.jpeg'):
        print(p)
        image = tf.keras.preprocessing.image.load_img(p)

        # image = adjust_contrast(contrast, image)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.6)
        # return image
        out = os.path.join(p.parent.parent.parent, 'contraste', p.parent.name)
        os.makedirs(out, exist_ok=True)
        fname = os.path.join(out, p.name)
        # print(fname)
        image.save(fname)

        # break