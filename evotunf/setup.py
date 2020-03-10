from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize

import os.path
import numpy as np


PACKAGE_NAME = 'evotunf'

cy_extensions = [
    Extension(
        'evotunf_ext',
        list(map(lambda f: os.path.join(PACKAGE_NAME, 'ext', f), ['__init__.pyx', 'evolutionary_tune_cpu.c'])),
        include_dirs=[np.get_include()],
        extra_compile_args=['-O2', '-fopenmp'],
        extra_link_args=['-fopenmp']
    )
]

setup(
    name=PACKAGE_NAME,
    packages=find_packages(),
    install_requires=['numpy', 'Cython'],
    ext_modules=cythonize(cy_extensions)
)
