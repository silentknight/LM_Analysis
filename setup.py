from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("speedup.pyx", annotate=True),
    include_dirs=[numpy.get_include()]
)