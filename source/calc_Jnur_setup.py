from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

#include_mpf_dir = "/u/rcooke/local/include/"
#lib_mpf_dir = "/u/rcooke/local/lib/"

ext = Extension("calc_Jnur", ["calc_Jnur.pyx"],
    include_dirs=[numpy.get_include()]
)

setup(ext_modules=[ext],
        cmdclass = {'build_ext': build_ext})

