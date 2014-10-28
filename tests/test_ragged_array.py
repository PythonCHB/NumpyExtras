#!/usr/bin/env python

"""
test_ragged_array.py

tests ofthe ragged_array class

designed to be run with nose
"""

import nose
import numpy as np

## add to path -- should be neccesary one there is a standard package
#import sys
#sys.path.append("../lib")


from numpy_extras.ragged_array import ragged_array as ra

def test_one_row():
    row = (1,2,3)
    a = ra([row,])
    assert np.array_equal(a[0], row)
    
def test_three_rows():
    rows = ([1,2,3],
            [4,5,6,7,8],
            [9, 10] )
    a = ra(rows)
    assert np.array_equal(a[0], rows[0])
    assert np.array_equal(a[1], rows[1])
    assert np.array_equal(a[2], rows[2])

def test_slice():    
    " a slice in the middle"    
    rows = ([1,2,3],
            [4,5,6,7,8],
            [9, 10],
            [11, 12, 13, 14, 15] )
    a = ra(rows)
    slice = a[1:3]
    print slice
    assert np.array_equal(slice[0], rows[1])
    assert np.array_equal(slice[1], rows[2])

def test_slice2():    
    " a slice with the first item"    
    rows = ([1,2,3],
            [4,5,6,7,8],
            [9, 10],
            [11, 12, 13, 14, 15] )
    a = ra(rows)
    slice = a[0:3]
    print slice
    assert np.array_equal(slice[0], rows[0])
    assert np.array_equal(slice[1], rows[1])
    assert np.array_equal(slice[2], rows[2])

def test_slice3():
    " a slice with the last item"    
    rows = ([1,2,3],
            [4,5,6,7,8],
            [9, 10],
            [11, 12, 13, 14, 15] )
    a = ra(rows)
    slice = a[2:4]
    print slice
    assert np.array_equal(slice[0], rows[2])
    assert np.array_equal(slice[1], rows[3])

def test_slice3():
    " a slice going over the end"
    rows = ([1,2,3],
            [4,5,6,7,8],
            [9, 10],
            [11, 12, 13, 14, 15] )
    a = ra(rows)
    slice = a[2:6]
    assert np.array_equal(slice[0], rows[2])
    assert np.array_equal(slice[1], rows[3])

def test_iteration():
    rows = ([1,2,3],
            [4,5,6,7,8],
            [9, 10],
            [11, 12, 13, 14, 15] )
    arr = ra(rows)
    for a, b in zip(rows, arr):
        assert np.array_equal(a, b)
    
def test_dtype():
    row = (1,2,3)
    a = ra([row,], dtype=np.float32)
    print a[0].dtype
    assert (a[0].dtype == np.float32)

def test_mixed_types():
    rows = ([1,2,3],
            [4.1,5.2,6.3,7.4,8.5],
            )
    a = ra(rows, dtype = np.float64)
    print a
    assert np.array_equal(a[0], rows[0]) 
    assert np.array_equal(a[1], rows[1]) 
    
def test_mixed_types2():
    rows = ([1,2,3],
            [4.1,5.2,6.3,7.4,8.5],
            )
    a = ra(rows)
    print a
    assert (a[0].dtype == np.float64)
    
        
def test_str():
    rows = ([1,2,3],
            [4,5,6,7,8],
            [9, 10] )
    a = ra(rows)
    rslt = "[1 2 3]\n[4 5 6 7 8]\n[ 9 10]\n"
    print "the str:",  str(a)
    assert (str(a) == rslt)

def test_repr():
    rows = ([1,2,3],
            [4,5,6,7,8],
            [9, 10] )
    a = ra(rows)
    rslt = "ragged array:\n[1 2 3]\n[4 5 6 7 8]\n[ 9 10]\n"
    print "the repr:",  repr(a)
    assert (repr(a) == rslt)
