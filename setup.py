import numpy as np

from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize


ext_modules = [
    Extension(
        r'a',
        [r'a.pyx']
    ),
]


setup(
    name='fast_load',
    version='1.0',
    url='https://github.com/xaaaandao/piperaceae-identification-paper',
    license='MIT',
    author='xandao',
    author_email='alexandre.ykz@gmail.com',
    description='',
    ext_modules=cythonize(ext_modules),
    include_dirs=[np.get_include()]
)
