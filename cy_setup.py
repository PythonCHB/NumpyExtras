#!/usr/bin/env python

"""
setup.py for building the cython parts:

cy_accumulator
"""

import numpy # to get the includes
  
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("numpy_extras.cy_accumulator",
                         ["cython_src/cy_accumulator.pyx"])]

setup(
    name = 'Accumulator',
    version = "0.1.0",
    description='A numpy array that can be used to accumulate data',
    author='Chris Barker',
    author_email='Chris.Barker@noaa.gov',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    include_dirs = [numpy.get_include(),],
    packages = ["numpy_extras"],
)


