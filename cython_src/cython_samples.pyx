cimport cython as cnp

def f(cython.floating[:] array):
    ...

f[cython.double](my_doubl_array)
f[cython.float](my_float_array)

cdef class C[cython.floating]:
    cdef cython.floating *c_array

C[double]()



cdef class list(object):
    cdef char *buf
    cdef int _case

    def __init__(self, dtype):
        # check what case dtype is (== np.int32)
        if dtype == np.uint8:
            self._case = cnp.NPY_UINT8


    cpdef append(self, x):
        cdef cnp.ndarray x_as_arr
        # resizing logic goes here, use memcpy
        if self._case == cnp.NPY_UINT8:
            (<np.uint8_t*>self.buf)[self.elem_count] = <np.uint8_t>x
        elif self._case == ...:
            (<np.int8_t*>self.buf)[self.elem_count] = <np.int8_t>x
        elif self._case == cnp.NPY_VOID:
            x_as_arr = x
            cnp.PyArray_ITEMSIZE(x_as_arr)
            cnp.PyArray_DATA(x_as_arr)
            (self.buf + itemsize * self.elem_count)


