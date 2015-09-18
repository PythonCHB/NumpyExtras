#!/usr/bin/env python

try:
    from setuptools import setup, Extension
except ImportError:
    print "You need setuptools to build this module"

from Cython.Build import cythonize

import numpy as np #for the include dirs...
#import os, sys

include_dirs = [np.get_include()]

ext_modules = cythonize(Extension("numpy_extras.filescanner",
                                  ["numpy_extras/filescanner.pyx"],
                                  include_dirs = include_dirs,
                                  language = "c",
                                 ))

# This setup is suitable for "python setup.py develop".
setup(name = "numpy_extras",
      version = "0.2.0",
      description='A few extra things for numpy',
      author='Chris Barker',
      author_email='Chris.Barker@noaa.gov',
      url="https://github.com/NOAA-ORR-ERD",
      license = "Public Domain",
      packages = ["numpy_extras"],
      ext_modules = ext_modules,
     classifiers=["Development Status :: 2 - Pre-Alpha",
                  "License :: Public Domain",
                  "Intended Audience :: Developers",
                  "Intended Audience :: Science/Research",
                  "Operating System :: OS Independent",
                  "Programming Language :: Cython",
                  "Programming Language :: Python :: 2 :: Only",
                  "Programming Language :: Python :: Implementation :: CPython",
                  "Topic :: Utilities",
                  "Topic :: Scientific/Engineering",
                 ],
    )

