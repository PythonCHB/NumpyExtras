NumpyExtras
===========

Same Add ons for numpy -- not very fully developed.

PUttin git up here in case it's useful to someone.

Accumulator
-----------

Accumulator is a numpy array class that can grow in place -- it uses a simple pre-alocate some extra space approach.

It turns out that it's about teh same speed as acumulating aveyting in a pyton list, and then converting it to a array, but it uses pess memory,an ddoes work beter with numpy special data types.

There is also an incmplete Cython version -- should be faster, but not robust enough for me to have fully tested perfromance.

Ragged Array
------------

A numpy ragged array implimentaiton -- not done, and I honeslty don't remember how far I got with it...


