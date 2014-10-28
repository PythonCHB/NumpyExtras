#!/usr/bin/env python

"""
some code to help profile the accumulator class

designed to be run with ipython "timeit"

"""

import numpy as np

import accumulator
# to make sure I'm getting a fresh copy while testing...
reload( accumulator)
from accumulator import accumulator


def list1(n):
    """
    using a list to accumulate integers, then make an array out of it.
    """
    l = []
    for i in xrange(n):
        l.append(i)
    return np.array(l)

def accum1(n):
    """
    using an accumulator to accumulate integers, then make an array out of it.
    """
    l = accumulator((), dtype=np.int)
    for i in xrange(n):
        l.append(i)
    return np.asarray(l)

def accum2(n):
    """
    Using an accumulator to accumulate integers, then make an array out of it.
    This time pre-allocating enough data
    """
    l = accumulator((), dtype=np.int)
    l.resize(n)
    for i in xrange(n):
        l.append(i)
    return np.asarray(l)
    
def array(n):
    """
    using an array, completely pre-allocated
    """
    l = np.empty((n,), dtype=np.int)
    for i in xrange(n):
        l[i] = i
    return np.asarray(l)
    
## now some tests putting in numpy data that's already numpy data:

def list_extend1(n):
    """
    using a list to built it up, then convert to a numpy array
    """
    l = []
    data = range(100)
    for i in xrange(n/10):
        l.extend(data)
    return np.array(l)
    
def accum_extend1(n):
    """
    Building it up with numpy data
    """
    l = accumulator((), dtype=np.int)
    data = np.arange(100, dtype=np.int)
    for i in xrange(n/10):
        l.extend(data)
    return np.asarray(l)
    

