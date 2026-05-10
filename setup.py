from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("speedup.pyx", annotate=True),
    include_dirs=[numpy.get_include()]
)