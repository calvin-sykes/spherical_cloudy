from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import Cython.Compiler.Options
import numpy

Cython.Compiler.Options.annotate = True

ext = Extension("cython_halo", ["cython_halo.pyx"],
                include_dirs=[numpy.get_include()],
                extra_compile_args=['-march=native'])

setup(ext_modules=[ext],
      cmdclass = {'build_ext': build_ext})

ext = Extension("cython_fns", ["cython_fns.pyx"],
                include_dirs=[numpy.get_include()],
                extra_compile_args=['-fopenmp', '-march=native'],
                extra_link_args=['-fopenmp'])

setup(ext_modules=[ext],
      cmdclass = {'build_ext': build_ext})
