#!/usr/bin/env python

"""
some code to help profile the Accumulator class

designed to be run with ipython "timeit"

One test run::

In [11]: run profile_accumulator.py


In [12]: timeit accum1(10000)

100 loops, best of 3: 3.91 ms per loop

In [13]: timeit list1(10000)

1000 loops, best of 3: 1.15 ms per loop

These are simply appending 10,000 integers in a loop -- with teh list, the list is turned into a numpy array at the end. So it's still faster to accumulate in a list, then make an array, but only a about a factor of 3 -- I think this is because you are staring with a python integer -- with the accumulator function, you need to be checking type and pulling a native integer out with each append. but a list can append a python object with no type checking or anything.

Then the conversion from list to array is all in C.

Note that the accumulator version is still more memory efficient...

In [14]: timeit accum2(10000)

100 loops, best of 3: 3.84 ms per loop

this version pre-allocated the whole internal buffer -- not much faster the buffer re-allocation isn't a big deal (thanks to ndarray.resize using realloc(), and not creating a new numpy array)

In [24]: timeit list_extend1(100000)

100 loops, best of 3: 4.15 ms per loop

In [25]: timeit accum_extend1(100000)

1000 loops, best of 3: 1.37 ms per loop

This time, the stuff is added in chunks 100 elements at a time -- the chunks being created ahead of time -- a list with range() the first time, and an array with arange() the second. much faster to extend with arrays...


"""

import numpy as np

import numpy_extras.accumulator
reload(numpy_extras.accumulator) # make sure to get a fresh copy.
Accumulator = numpy_extras.accumulator.Accumulator


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
    using an Accumulator to accumulate integers, then make an array out of it.
    """
    l = Accumulator((), dtype=np.int)
    for i in xrange(n):
        l.append(i)
    return np.asarray(l)

def accum2(n):
    """
    Using an Accumulator to accumulate integers, then make an array out of it.
    This time pre-allocating enough data
    """
    l = Accumulator((), dtype=np.int)
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
    num_to_extend = 100
    data = range(num_to_extend)
    for i in xrange(n/num_to_extend):
        l.extend(data)
    return np.array(l)
    
def accum_extend1(n):
    """
    Building it up with numpy data
    """
    num_to_extend = 100
    l = Accumulator((), dtype=np.int)
    data = np.arange(num_to_extend, dtype=np.int)
    for i in xrange(n/num_to_extend):
        l.extend(data)
    return np.asarray(l)
    

