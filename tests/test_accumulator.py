#!/usr/bin/env python

"""
tests for the Accumulator class

designed to be run with nose
"""

import unittest
import numpy as np
from numpy.testing import assert_array_equal
from numpy_extras.accumulator import Accumulator

class test_init(unittest.TestCase):
    
    def test_nd(self):
        self.assertRaises(ValueError, Accumulator,  ((1,2),(3,4) )  )

    def test_empty(self):
        a = Accumulator()
        self.assertEqual( len(a), 0 )
        
    def test_simple(self):
        a = Accumulator( (1,2,3) )
        self.assertEqual( len(a), 3 )
        
    def test_buffer_init(self):
        a = Accumulator()
        self.assertEqual( a.buffersize, a.DEFAULT_BUFFER_SIZE )

    def test_dtype(self):
        a = Accumulator(dtype=np.uint8)
        self.assertEqual( a.dtype, np.uint8 )

    def test_shape(self):
        a = Accumulator((1,2,4.5), dtype=np.float64)
        self.assertEqual( a.shape, (3,) )
        
    def test_scalar(self):
        """
        passing a scalar in to the __init__ should give you a length-one array,
        as it doesn'tmake sesne to have a scalar Accumulator
        """
        a = Accumulator(5)
        self.assertEqual(len(a), 1)
        
class test_indexing(unittest.TestCase):
    
    def test_simple_index(self):
        a = Accumulator( (1,2,3,4,5) )
        self.assertEqual(a[1], 2)

    def test_neg_index(self):
        a = Accumulator( (1,2,3,4,5) )
        self.assertEqual(a[-1], 5)

    def test_index_too_big(self):
        a = Accumulator( (1,2,3,4,5) )
        # I can't figure out how to use asserRaises for a non-callable
        try:
            a[6]
        except IndexError:
            pass
        else:
            raise Exception("This test didn't raise an IndexError")

    def test_append_then_index(self):
        a = Accumulator( () )
        for i in range(20):
            a.append(i)
        self.assertEqual(a[15], 15)

    def test_indexs_then_resize(self):
       """
       this here to see if having a view on teh buffer causes problems
       """
       a = Accumulator( (1,2,3,4,5) )
       b = a[4]
       a.resize(1000)


class test_slicing(unittest.TestCase):
    
    def test_simple_slice(self):
        a = Accumulator( (1,2,3,4,5) )
        assert_array_equal(a[1:3], np.array([2, 3]))

    def test_too_big_slice(self):
        b = np.array( (1.0, 3, 4, 5, 6) )
        a = Accumulator( b )
        assert_array_equal(a[1:10], b[1:10])

    def test_full_slice(self):
        b = np.array( (1.0, 3, 4, 5, 6) )
        a = Accumulator( b )
        assert_array_equal(a[:], b[:])

    def test_slice_then_resize(self):
       """
       this here to see if having a view on teh buffer causes problems
       """
       a = Accumulator( (1,2,3,4,5) )
       b = a[2:4]
       a.resize(1000)


class test_append(unittest.TestCase):
    
    def test_append_length(self):
        a = Accumulator( (1,2,3) )
        a.append(4)
        self.assertEqual(len(a), 4)

class test_extend(unittest.TestCase):
    
    def test_extend_length(self):
        a = Accumulator( (1,2,3) )
        a.extend( (4, 5, 6) )
        self.assertEqual(len(a), 6)

    def test_extend_long(self):
        a = Accumulator( range(100) )
        a.extend( range(100) )
        self.assertEqual(a.length, 200)


class test_resize(unittest.TestCase):
    
    def test_resize_longer(self):
        a = Accumulator( (1,2,3) )
        a.resize(1000)
        self.assertEqual(a.buffersize, 1000)

    def test_resize_too_short(self):
        a = Accumulator( (1,2,3,4,5,6,7,8) )
        self.assertRaises(ValueError, a.resize, 5)
        
    def test_fitbuffer(self):
        a = Accumulator( (1,2,3) )
        a.fitbuffer()
        self.assertEqual(a.buffersize, 3)


class test__array__(unittest.TestCase):
    
    def test_asarray(self):
        a = Accumulator( (1,2,3) )
        b = np.asarray(a)
        print b
        assert_array_equal(a[:], b)
    
    def test_asarray_dtype(self):
        a = Accumulator( (1,2,3), dtype=np.uint32 )
        b = np.asarray(a, dtype=np.float)
        self.assertEqual(b.dtype, np.float)
    
class test_strings(unittest.TestCase):

    def test_str(self):
        a = Accumulator( (1,2,3), dtype=np.float )
        self.assertEqual(str(a), '[ 1.  2.  3.]')

    def test_repr(self):
        a = Accumulator( (1,2,3), dtype=np.float )
        self.assertEqual(repr(a), 'Accumulator([ 1.,  2.,  3.])')




class test_complex_data_types(unittest.TestCase):
    # tests making a 1-d array of a complex numpy dtype
    dt = np.dtype([('f0', '<i4'), ('f1', '<f8', (2, 3))])
    item = (5, [[3.2, 4.5, 6.5], [1.2, 4.2e7, -45.76]])

    def test_dtype1(self):
        a = Accumulator( self.item, dtype=self.dt )
        self.assertEqual(a[0][0], 5)

    def test_dtype2(self):
        a = Accumulator( self.item, dtype=self.dt )
        assert_array_equal(a[0][1],
                           np.array([[3.2, 4.5, 6.5], [1.2, 4.2e7, -45.76]], dtype=np.float64)
                           )
                           
    def test_dtype3(self):
        a = Accumulator(dtype=self.dt )
        for i in range(100):
            a.append(self.item)
        assert_array_equal( a[99][1][1,2], -45.76 )
        
        