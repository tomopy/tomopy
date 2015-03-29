============
User's Guide
============

:Page Status: Incomplete
:Last Reviewed: 2015-03-29


.. contents:: Contents
   :local:


Data Structures
===============

In TomoPy, data is stored in 
`Numpy <http://docs.scipy.org/doc/numpy/user/>`_ arrays, which can be
efficiently used to represent and manupilate large data sets.

Input data types
----------------

Although most cameras capture images in 16-bit dynamic range, many 
functions in the package support 32-bit float precision data for 
processing. So, you should ALWAYS convert the raw data into 
``float32`` before calling a function. You can simply do this by
using ``data_as_float32`` function in ``io`` module::

    from tomopy.io.data import *
    import numpy as np

    dat = np.array([0, 1, 2], dtype='int')
    dat = as_float32(dat)

This will create a one to one conversion of int to float.

The built-in functions for data importing have the ``dtype`` argument
to specify the type of the output data for convenience. For example::

     dat = lena(dtype='float32')

If we summarize the key points:

1. All input data must be Numpy arrays;

2. All data types must be converted to ``float32`` before processing.

.. warning:: Currently 64-bit operations are not supported, so DO NOT 
    use ``astype='float'``, which by default creates data in 64-bit 
    precision.


Output data types
-----------------

Output data is almost always in ``float32`` precision, and can be 
converted to 8-bit or 16-bit integer to save some space. See the 
following example to convert an array into ``uint16``::

    dat = np.arange(0, 1, 0.1).astype('float32')
    dat = as_uint16(dat, dmin=0.1, dmax=0.8)

You can use ``dmin`` and ``dmax`` arguments to set a desired scaling 
range. Otherwise the minimum and maximum values of the data will be 
used for integer conversion.


Loading and saving data
=======================

Data loading from and saving to various file formats are performed 
by the ``tomopy.io`` sub-package.

Loading from HDF5 files
-----------------------
Data can be imported from a group in an HDF5 file using::

    from tomopy.io import data
    mydata = data.read_hdf5(fname='path/mydata', gname='/exchange/data')


Loading built-in test data
--------------------------
Various built-in test data are available in ``tomopy.io.phantom`` module.
They can be loaded as follows::

    from tomopy.io import phantom 
    arr = phantom.lena()

The returned data are by default 3-D and in ``float32`` precision. You 
can also try ``baboon``, ``barbara``, ``cameraman``, ``lena``, ``peppers``, 
``shepp2d`` or ``shepp3d`` functions.


Saving as HDF5 files
---------------------
Data can be written to a group in an HDF5 file using::
 
    from tomopy.io import data
    data.write_hdf5(mydata, fname='path/mydata', gname='/exchange')


Saving as a TIFF stack
----------------------
A 3-D data can be written as a stack of TIFF images using::

    from tomopy.io import data
    data.write_tiff_stack(mydata, file_name='path/mydata', axis=0)

``axis`` argument determines the axis to be used for slicing the 3-D data
volume.

.. warning:: DO NOT use the file extension for saving data. It will be 
   automatically added depending on the called function. 


Data value corrections
======================

Negative values
---------------
In principle the measurement data should not contain any negative
values. However for some cases this is not true, and needs to 
be corrected. ``rm_negval`` function can be used to set 
these values to a specified value::

    dat = np.array([-1., 1., 2.], dtype='float32')
    dat = rm_negval(dat, val=0.)


NaN values
----------
Similar to the negative value correction, NaN values can be replaced 
by any specified value using ``rm_nan`` function::

    dat = np.array([-1., 1., np.nan], dtype='float32')
    dat = rm_nan(dat, val=0.)
