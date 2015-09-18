#!/usr/bin/env python

"""
ragged_array

A "ragged" array class -- build on numpy

The idea is to be able to store data that is essentially 2-d, but each row is
an arbitrary length, like:

1   2   3
4   5   6   7   8   9 
10 11  
12 13  14  15  16  17  18
19 20  21
...

This can also be extended to support higher dimensional arrays, as long as
only the last dimension is the "ragged" one.

Internally, the data is stored as an array of one less dimension, with indexes
into the array to identify the "rows". 

The array can be indexed by row.

Operations can be done on the entire array just like any numpy array
  - this is one of the primary advantages of using a single numpy array 
    for the internal storage.
  - Another advantage is that the data are all in a block that can be
    passed off and manipulated in C or Cython  
  - operations that require slicing, specifying an axis, etc, are likely to
    fail
"""

import numpy as np

class ragged_array:
    
    def __init__(self, data, dtype=None):
        """
        create a new ragged array
        
        data should be a sequence of sequences:
        [ [1, 2, 3], [4,5,6,7], [4,2] ]
        
        if no dtype is provided, it will be determined by the type of the first row

        """
        
        # generate the arrays:
        a = []
        ind = [0]
        # flatten the data sequence:
        for row in data:
            a.extend(row)
            ind.append(ind[-1]+len(row))
        self._data_array = np.array(a, dtype=dtype)
        # note: using "np.int" for the index array should give me 32 bits on 
        #       32 bit python, and 64 bits on 64 bit python.
        self._index_array = np.array(ind, dtype=np.int) 

    def append(self, row):

        """
        Should this be supported?
        
        It does require a copy of the data array
        
        """
        self._data_array = np.r_[self._data_array, row]
        self._index_array  = np.r_[self._index_array, (self._data_array.shape[0],) ]

        
    def __len__(self):
        return len(self._index_array) - 1 # there is an extra index at the end, so that IndexArray[i+1] works
        
    def __getitem__(self,index):
        """
        returns a numpy array of one row.
        """
        if index > (len(self._index_array) - 1):
            raise IndexError
        if  index < 0:
             if index < - (len(self._index_array) -1 ):
                 raise IndexError
             index = len(self._index_array) -1 + index
        row = (self._data_array[self._index_array[index]:self._index_array[index+1]] )
        return row

    def __getslice__(self, i, j):
        """
        ra.__getslice__(i, j) <==> a[i:j]
    
        Use of negative indices is not supported.
        
        This a view, just like numpy arrays
        """
        ## this builds a new ragged_array, as a view onto the original
        ##  fixme: this seems like it should be more elegant.
        j = min( j, len(self) )
        rslt = ragged_array(((),),)
        rslt._data_array = self._data_array[self._index_array[i]:self._index_array[j]]
        rslt._index_array = np.r_[0, (self._index_array[i+1:j+1] - self._index_array[i])]
        return rslt
    
    def __string_middle(self):
        '''
        helper function that generates a list of strings fr the rows in the array
        '''
        middle = []
        if len(self._data_array) > np.get_printoptions()['threshold']:
            for row in self[:3]:
                middle.append(str(row))
            middle.append("...")
            for row in self[-3:]:
                middle.append(str(row))
        else:
            for row in self:
                middle.append(str(row))
                
        return middle
    
    def __str__(self):
        """
        present a nice string representation of the array
        """
        msg = self.__string_middle()
        return "\n".join(msg)

    def __repr__(self):
        """
        present a nice string representation of the array that is "evaluateable" 
        """
        msg = ['ragged_array([']
        middle = self.__string_middle()
        middle = ",\n              ".join(middle)
        msg.append(middle)
        msg.append('])')
        msg.append("")
        return "".join(msg)
    
    def flat(self):
        """
        returns a flattend version of the array -- 1-d
        
        actually returns the internal array representation, so it shares a view with the ragged array
        
        """
        return self._data_array
            

