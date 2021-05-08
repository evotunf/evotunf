from setuptools import setup, find_packages
from distutils.extension import Extension
from distutils.command.build_ext import build_ext
from Cython.Build import cythonize

import os
import numpy as np


PACKAGE_NAME = 'evotunf'


def find_in_path(name, path=None):
    if not path:
        path = os.environ['PATH']
    for dir in path.split(os.pathsep):
        binpath = os.path.join(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = os.path.join(home, 'bin', 'nvcc')
    else:
        nvcc = find_in_path('nvcc')
        if nvcc is None:
            raise EnvironmentError(
                "The nvcc binary could not be located "
                "in your $PATH. Either add it to your path, or set $CUDAHOME.")
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {
        'home': home,
        'nvcc': nvcc,
        'include': os.path.join(home, 'include'),
        'lib64': os.path.join(home, 'lib64'),
    }

    for item, path in cudaconfig.items():
        if not os.path.exists(path):
            raise EnvironmentError("The CUDA {item} path could not be located in {path}.".format(
                item=item, path=path))
    return cudaconfig


CUDA = locate_cuda()

cy_extensions = [
    Extension(
        'evotunf_ext',
        list(map(lambda f: os.path.join(PACKAGE_NAME, 'ext', f), [
            '__init__.pyx',
            'evolutionary_tune_gpu.cu',
            'evolutionary_tune_cpu.c',
            'ga_params.c',
        ])),
        library_dirs=[CUDA['lib64']],
        libraries=['stdc++', 'cudart'],
        runtime_library_dirs=[CUDA['lib64']],
        include_dirs=[np.get_include(), CUDA['include']],
        extra_compile_args={
            'gcc': ['-g3', '-Og', '-fopenmp'],
            'nvcc': ['-arch=sm_35', '-lineinfo', '--maxrregcount=32', '--ptxas-options', '-O3,-v', '--compiler-options', '-fPIC']
        },
        extra_link_args=['-fopenmp', '-Og', '-g3'],
    )
]


def customize_compiler_for_nvcc(self):
    self.src_extensions.append('.cu')
    default_compiler_so = self.compiler_so
    super = self._compile

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        _, ext = os.path.splitext(src)
        if ext == '.cu':
            self.set_executable('compiler_so', CUDA['nvcc'])
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']
        super(obj, src, ext, cc_args, postargs, pp_opts)
        self.compiler_so = default_compiler_so

    self._compile = _compile


class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

setup(
    name=PACKAGE_NAME,
    packages=[PACKAGE_NAME],
    install_requires=['numpy', 'Cython'],
    ext_modules=cythonize(cy_extensions),
    cmdclass={'build_ext': custom_build_ext}
)
