from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import distutils.sysconfig as ds

# Anaconda ships an old compiler_compat/ld that rejects Ubuntu 22.04+ libc,
# which uses the RELR ELF section format (.relr.dyn) requiring binutils >= 2.26.
# Strip the -B .../compiler_compat flag so gcc uses the system linker instead.
_cfg = ds.get_config_vars()
for _key in ('LDSHARED', 'BLDSHARED'):
    if _key not in _cfg:
        continue
    _parts = _cfg[_key].split()
    _out, _skip = [], False
    for _p in _parts:
        if _skip:
            _skip = False
            continue
        if _p == '-B':
            _skip = True
            continue
        if _p.startswith('-B') and 'compiler_compat' in _p:
            continue
        _out.append(_p)
    _cfg[_key] = ' '.join(_out)

setup(
    ext_modules=cythonize(
        "speedup.pyx",
        annotate=True,
        compiler_directives={"language_level": "3"},
    ),
    include_dirs=[numpy.get_include()],
)
