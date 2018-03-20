from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import Cython.Compiler.Options
import numpy

#include_mpf_dir = "/u/rcooke/local/include/"
#lib_mpf_dir = "/u/rcooke/local/lib/"

Cython.Compiler.Options.annotate = True

ext = Extension("cython_fns", ["cython_fns.pyx"],
    include_dirs=[numpy.get_include()]
)

setup(ext_modules=[ext],
        cmdclass = {'build_ext': build_ext})

