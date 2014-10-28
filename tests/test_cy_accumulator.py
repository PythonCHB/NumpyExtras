#!/usr/bin/env python

"""
tests for the cython accumulator class

designed to be run with pytest (though there is a bunch of legacy pyunit
stuff in here

It may well run with other test runners, like nose.
"""


import pytest
import unittest # for the legacy stuff -- should clean this out...
import numpy as np
from numpy.testing import assert_array_equal
from numpy_extras.cy_accumulator import Accumulator

print "cy_accumulator successfully imported"

class test_startup(unittest.TestCase):

    def test_empty(self):
        a = Accumulator()
        print len(a)
        a.append(4)
        print len(a)
        self.assertEqual(len(a), 1)

    def test_scalar(self):
        a = Accumulator(3)
        self.assertEqual(len(a), 1)
        assert_array_equal(np.array(a), (3,))

    def test_uint8(self):
        """
        tests using the uint8 type
        """
        a = Accumulator( (1,2,3), dtype=np.uint8 )
        print a
        
        assert np.array_equal(a, np.array((1,2,3), dtype=np.uint8) )

    def test_int32(self):
        print "running test_int32"
        a = Accumulator( (1,2,3), dtype=np.int32 )
        a.append(4)
        self.assertEqual(len(a), 4)


    ## fixme: auto-generate these?
    ## these are all types that my numpy doesn't support
#    def test_unsupported_type1(self):
#        self.assertRaises(NotImplementedError, Accumulator, (1,2,3), dtype=np.int96  )
#    def test_unsupported_type2(self):
#        self.assertRaises(NotImplementedError, Accumulator, (1,2,3), dtype=np.int256  )
#    def test_unsupported_type3(self):
#        self.assertRaises(NotImplementedError, Accumulator, (1,2,3), dtype=np.uint96  )
#    def test_unsupported_type4(self):
#        self.assertRaises(NotImplementedError, Accumulator, (1,2,3), dtype=np.uint256  )
#    def test_unsupported_type5(self):
#        self.assertRaises(NotImplementedError, Accumulator, (1,2,3), dtype=np.float16  )
#    def test_unsupported_type6(self):
#        self.assertRaises(NotImplementedError, Accumulator, (1,2,3), dtype=np.float256  )

#    def test_unsupported_type5(self):
#        dt = np.dtype([('x','f4'),('y','f4')])
#        a = np.empty((1,), dtype=dt)
#        a[0] = (3.0, 5.0)
#        print dt, a
#        self.assertRaises(NotImplementedError, Accumulator, (1,2,3), dtype=dt )


    def test_to_array(self):
        """
        a test assured to fail to get output, etc
        """
        a = Accumulator( (1,2,3), dtype=np.uint8 )

        arr = np.array(a)
        assert arr.strides == (1,)
        assert arr.dtype == np.uint8
        assert arr.shape == (3,)
    
    def test_dummy(self):
        """
        a test assured to fail to get output, etc
        """
        a = Accumulator( (1,2,3), dtype=np.int32 )

        arr = np.array(a)
        print arr
        print arr.strides
        print arr.dtype
#        a.append(4)
#        print "length is now:", len(a)

    def test_delete(self):
        a = Accumulator( (1,2,3), dtype=np.int32 )
        del a
        ## fixme -- how do I check to see if the memory was freed?
        assert True
        
            
class test_append(unittest.TestCase):
    def test_append_length_empty(self):
        a = Accumulator( (), )
        a.append(4)
        a.append(5)
        self.assertEqual(len(a), 2)

    def test_append_length(self):
        a = Accumulator( (1,2,3) )
        a.append(4)
        a.append(5)
        self.assertEqual(len(a), 5)

    def test_append_length_middle(self):
        """
        this should test that re-allocation is working for just enough for a re-allocate
        """
        a = Accumulator( (2,3,4) )
        for i in xrange(15):
            a.append(i)
        self.assertEqual(len(a), 18)

    def test_append_length_big(self):
        """
        this should test that re-allocation is working for larger data
        """
        a = Accumulator( (2,3,4) )
        for i in xrange(10000):
            a.append(i)
        self.assertEqual(len(a), 10003)

    def test_append_unsupported_type(self):
        a = Accumulator( (1,2,3), dtype=np.int32 )
        self.assertRaises(TypeError, a.append, "this")

## a test of dtypes..
@pytest.mark.parametrize( ('dt', 'data'),
                          [ (np.int8,     (1,-2,3) ),
                            (np.int16,    (1,-2,3) ),
                            (np.int32,    (1,-2,3) ),
                            (np.int64,    (1,-2,3) ),
                            (np.uint8,    (1,2,3) ),
                            (np.uint16,   (1,2,3) ),
                            (np.uint32,   (1,2,3) ),
                            (np.uint64,   (1,2,3) ),
                            (np.float32,   (1.0,-2.2,3.3) ),
                            (np.float64,   (1.0,-2.2,3.3) ),
                            (np.complex64,   (1.0 + 2.0j, -2.2+ 3.3j, 3.3-4.4j) ),
                            (np.complex128,   (1.0 + 2.0j, -2.2+ 3.3j, 3.3-4.4j)  ),
                            ])
def test_data_type(dt, data):
    a = Accumulator( dtype=dt )
    for x in data:
        a.append(x)
    assert a.dtype == dt
    assert np.array_equal(a, np.array(data, dtype=dt) )

    

class test_errors(unittest.TestCase):
    
    def test_set_dtype(self):
        a = Accumulator( (1,2,3), dtype=np.int32 )
        #setattr(a, "dtype", np.float32)
        self.assertRaises(AttributeError, setattr, a, "dtype", np.float32)
    

#class test_init(unittest.TestCase):
#    
#    def test_nd(self):
#        self.assertRaises(ValueError, Accumulator,  ((1,2),(3,4) )  )
#
#    def test_empty(self):
#        a = Accumulator()
#        self.assertEqual( len(a), 0 )
#        
#    def test_simple(self):
#        a = Accumulator( (1,2,3) )
#        self.assertEqual( len(a), 3 )
#        
##    def test_buffer_init(self):
##        a = Accumulator()
##        self.assertEqual( a.buffersize, a.DEFAULT_BUFFER_SIZE )
#
#    def test_dtype(self):
#        a = Accumulator(dtype=np.uint8)
#        self.assertEqual( a.dtype, np.uint8 )
#
#    def test_shape(self):
#        a = Accumulator((1,2,4.5), dtype=np.float64)
#        self.assertEqual( a.shape, (3,) )
#        
#    def test_scalar(self):
#        """
#        passing a scalar in to the __init__ should give you a length-one array,
#        as it doesn't make sense to have a scalar Accumulator
#        """
#        a = Accumulator(5)
#        self.assertEqual(len(a), 1)
#        
def test_index_out_of_bounds():
    a = Accumulator( (1,2,3,4,5) )
    with pytest.raises(IndexError):
        a[32]
    with pytest.raises(IndexError):
        a[5]
    with pytest.raises(IndexError):
        a[-6]
        
def test_slice():
    """
    this should work at some point, but not yet...
    """
    a = Accumulator( (1,2,3,4,5) )
#    with pytest.raises(NotImplementedError):
#        a[1:3]
    print a[1:3]
    print a[1:3:2]
    assert False

def test_simple_index():
    a = Accumulator( (1,2,3,4,5) )
    print a[0],
    print a[1],
    print a[2],
    print a[-1]
    print a[-2]
    assert a[0] == 1
    assert a[1] == 2
    assert a[2] == 3
    assert a[3] == 4
    assert a[4] == 5
    assert a[-1] == 5
    assert a[-2] == 4
    assert a[-3] == 3
    assert a[-4] == 2
    assert a[-5] == 1

## a test of indexing with various dtypes..
@pytest.mark.parametrize( ('dt', 'data'),
                          [ (np.int8,     (1,-2,3) ),
                            (np.int16,    (1,-2,3) ),
                            (np.int32,    (1,-2,3) ),
                            (np.int64,    (1,-2,3) ),
                            (np.uint8,    (1,2,3) ),
                            (np.uint16,   (1,2,3) ),
                            (np.uint32,   (1,2,3) ),
                            (np.uint64,   (1,2,3) ),
                            (np.float32,   (1.0,-2.2,3.3) ),
                            (np.float64,   (1.0,-2.2,3.3) ),
                            (np.complex64,   (1.0 + 2.0j, -2.2+ 3.3j, 3.3-4.4j) ),
                            (np.complex128,   (1.0 + 2.0j, -2.2+ 3.3j, 3.3-4.4j)  ),
                            ])
def test__index_data_type(dt, data):
    a = Accumulator( data, dtype=dt )
    for i in range(len(data)):
        assert np.allclose(a[i], data[i])



#    def test_neg_index(self):
#        a = Accumulator( (1,2,3,4,5) )
#        self.assertEqual(a[-1], 5)
#
#    def test_index_too_big(self):
#        a = Accumulator( (1,2,3,4,5) )
#        # I can't figure out how to use assertRaises for a non-callable
#        try:
#            a[6]
#        except IndexError:
#            pass
#        else:
#            raise Exception("This test didn't raise an IndexError")
#
#    def test_append_then_index(self):
#        a = Accumulator( () )
#        for i in range(20):
#            a.append(i)
#        self.assertEqual(a[15], 15)
#
#    def test_indexs_then_resize(self):
#       """
#       this here to see if having a view on the buffer causes problems
#       """
#       a = Accumulator( (1,2,3,4,5) )
#       b = a[4]
#       a.resize(1000)
#
#
#class test_slicing(unittest.TestCase):
#    
#    def test_simple_slice(self):
#        a = Accumulator( (1,2,3,4,5) )
#        assert_array_equal(a[1:3], np.array([2, 3]))
#
#    def test_too_big_slice(self):
#        b = np.array( (1.0, 3, 4, 5, 6) )
#        a = Accumulator( b )
#        assert_array_equal(a[1:10], b[1:10])
#
#    def test_full_slice(self):
#        b = np.array( (1.0, 3, 4, 5, 6) )
#        a = Accumulator( b )
#        assert_array_equal(a[:], b[:])
#
#    def test_slice_then_resize(self):
#       """
#       this here to see if having a view on the buffer causes problems
#       """
#       a = Accumulator( (1,2,3,4,5) )
#       b = a[2:4]
#       a.resize(1000)
#
#
#class test_append(unittest.TestCase):
#    
#    def test_append_length(self):
#        a = Accumulator( (1,2,3) )
#        a.append(4)
#        self.assertEqual(len(a), 4)
#
#class test_extend(unittest.TestCase):
#    
#    def test_extend_length(self):
#        a = Accumulator( (1,2,3) )
#        a.extend( (4, 5, 6) )
#        self.assertEqual(len(a), 6)
#
#    def test_extend_long(self):
#        a = Accumulator( range(100) )
#        a.extend( range(100) )
#        self.assertEqual(a.length, 200)
#
#
#class test_resize(unittest.TestCase):
#    
#    def test_resize_longer(self):
#        a = Accumulator( (1,2,3) )
#        a.resize(1000)
#        self.assertEqual(a.buffersize, 1000)
#
#    def test_resize_too_short(self):
#        a = Accumulator( (1,2,3,4,5,6,7,8) )
#        self.assertRaises(ValueError, a.resize, 5)
#        
#    def test_fitbuffer(self):
#        a = Accumulator( (1,2,3) )
#        a.fitbuffer()
#        self.assertEqual(a.buffersize, 3)
#
#
#class test__array__(unittest.TestCase):
#    
#    def test_asarray(self):
#        a = Accumulator( (1,2,3) )
#        b = np.asarray(a)
#        print b
#        assert_array_equal(a[:], b)
#    
#    def test_asarray_dtype(self):
#        a = Accumulator( (1,2,3), dtype=np.uint32 )
#        b = np.asarray(a, dtype=np.float)
#        self.assertEqual(b.dtype, np.float)
#    
#class test_strings(unittest.TestCase):
#
#    def test_str(self):
#        a = Accumulator( (1,2,3), dtype=np.float )
#        self.assertEqual(str(a), '[ 1.  2.  3.]')
#
#    def test_repr(self):
#        a = Accumulator( (1,2,3), dtype=np.float )
#        self.assertEqual(repr(a), 'Accumulator([ 1.,  2.,  3.])')
#
#
#class test_complex_data_types(unittest.TestCase):
#    dt = np.dtype([('f0', '<i4'), ('f1', '<f8', (2, 3))])
#    item = (5, [[3.2, 4.5, 6.5], [1.2, 4.2e7, -45.76]])
#
#    def test_dtype1(self):
#        a = Accumulator( self.item, dtype=self.dt )
#        self.assertEqual(a[0][0], 5)
#
#    def test_dtype2(self):
#        a = Accumulator( self.item, dtype=self.dt )
#        assert_array_equal(a[0][1],
#                           np.array([[3.2, 4.5, 6.5], [1.2, 4.2e7, -45.76]], dtype=np.float64)
#                           )
#                           
#    def test_dtype3(self):
#        a = Accumulator(dtype=self.dt )
#        for i in range(100):
#            a.append(self.item)
#        assert_array_equal( a[99][1][1,2], -45.76 )
#        
        