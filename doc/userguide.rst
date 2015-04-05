============
User's guide
============

This section explains the basics of TomoPy usage.

.. contents:: Contents
   :local:


Data I/O
========

Data structures
---------------

In TomoPy, data is stored in 
`Numpy <http://docs.scipy.org/doc/numpy/user/>`_ arrays, which can be
efficiently used to represent and manupilate large data sets.

Input data types
~~~~~~~~~~~~~~~~

Although most cameras capture images in 16-bit dynamic range, many 
functions in the package support 32-bit float precision data for 
processing. So, you should ALWAYS convert the raw data into 
``float32`` before calling a function. You can simply do this by
using ``as_float32`` function::
    
    >>> import tomopy
    >>> arr = range(0, 10)
    >>> arr = tomopy.as_float32(arr)

The built-in functions for data importing have the ``dtype`` argument
to specify the type of the output data for convenience. For example:

    >>> arr = tomopy.lena(dtype='float32')

If we summarize the key points, all input data must be Numpy arrays,
and all data types must be converted to ``float32`` before processing.

.. warning:: Currently 64-bit operations are not supported, so DO NOT 
    use ``astype='float'``, which by default creates data in 64-bit 
    precision.

Output data types
~~~~~~~~~~~~~~~~~

Output data is almost always in ``float32`` precision, and can be 
converted to 8-bit or 16-bit integer to save some space. See the 
following example to convert an array into ``uint16``::

    >>> import numpy as np
    >>> arr = np.arange(0, 1, 0.1).astype('float32')
    >>> arr = tomopy.as_uint16(arr, dmin=0.4, dmax=0.8)
    >>> print(arr)
    [    0     0     0     0     0 16383 32767 49151 65535 65535]

You can use ``dmin`` and ``dmax`` arguments to set a desired scaling 
range. Otherwise the minimum and maximum values of the data will be 
used for integer conversion.


Importing data
--------------

There are various common file formats for storing tomography data. 
Although the popular ones are HDF5 (Hierarchical Data Format) and TIFF 
(Tagged Image File Format), TomoPy allows to read data from other 
formats including SPE (Princeton Instruments), XRM (Xradia), 
EDF (European Data Format), and DM3 (Digital Micrograph). Here we 
briefly desctibe how one can read data from these formats:

Read from HDF5 file
~~~~~~~~~~~~~~~~~~~~

Data can be read from a group in an HDF5 file using::

    >>> arr = tomopy.read_hdf5(fname='path/mydata', gname='/exchange/data')

Read from TIFF stack
~~~~~~~~~~~~~~~~~~~~

Typically the projection images of a tomography experiment is stored
as a stack of 2-D TIFF files in a folder. The following function 
reads all TIFF stack in a folder and converts them to a 3-D Numpy array::

    >>> arr = tomopy.read_tiff_stack(fname='path/mydata')

Data recipes
~~~~~~~~~~~~

The concept of *recipes* is to facilitate data exchange among researchers. Because there are more than one ways to save a dataset in a single format,
it is hard to understand the conventions used to store others' datasets. 
For example one can use a different indexing convention to export individual projection TIFF files, or have multiple datasets in different groups
in a single HDF5 file. To ease this problem, we provide *recipes* 
for known data conventions. For example, data produced at the 
APS-2BM beamline can simply be imported without the need to know in 
which format it is stored::

    >>> data, white, dark = tomopy.read_aps2bm(fname='path/mydata')

Note that the output can be more than one data depending on the specific
experiment. 

Slicing data
~~~~~~~~~~~~

Sometimes, you may want to load a few slices or projections from the
complete data. *Slicing* allows you to do this. For example the 
following reads projections between 30 and 40 in 2 incremental steps::

    >>> arr = tomopy.read_aps2bm(fname='path/mydata', proj=slice(30, 40, 2))

Built-in test data
~~~~~~~~~~~~~~~~~~

There are a number of built-in data that can be used for testing
purposes. You can load them simply::

    >>> arr = tomopy.lena()

The returned data are by default 3-D and in ``float32`` precision.
Full list of available built-in data is presented below: 

.. image:: img/test-data.png


Exporting data
--------------

Write to HDF5 file
~~~~~~~~~~~~~~~~~~~~

Data can be written to a group in an HDF5 file using::

    >>> tomopy.write_hdf5(mydata, fname='path/mydata', gname='/exchange')

Write to TIFF stack
~~~~~~~~~~~~~~~~~~~~

A 3-D data can be written as a stack of TIFF images using::

    >>> tomopy.write_tiff_stack(mydata, file_name='path/mydata', axis=0)

The ``axis`` argument determines the axis to be used for slicing the 
3-D data volume.

.. warning:: DO NOT use the file extension for saving data. It will be 
   automatically added depending on the called function. 


Data corrections
================

Value checks
------------

Negative values
~~~~~~~~~~~~~~~

In principle the measurement data should not contain any negative
values. However for some cases this is not true, and needs to 
be corrected. ``remove_neg`` function can be used to set 
these values to a specified value::

    >>> arr = np.arange(-5, 5)
    >>> arr = tomopy.remove_neg(arr, val=0.)
    >>> print(arr)
    [0 0 0 0 0 0 1 2 3 4]

NaN values
~~~~~~~~~~

Similar to the negative value correction, NaN values can be replaced 
by any specified value using ``remove_nan`` function::

    >>> arr = np.array([-1., 1., np.nan])
    >>> arr = tomopy.remove_nan(arr, val=123.)
    >>> print(arr)
    [-1.  1.  123.]


Filtering
---------

Median filtering
~~~~~~~~~~~~~~~~

.. Todo:: Explain how median filter is applied. 

Zinger removal
~~~~~~~~~~~~~~

.. Todo:: Explain how zinger removal is applied. 

Stripe removal
~~~~~~~~~~~~~~

.. Todo:: Explain how stripe removal is applied. 


Phase retrieval
===============

Near-field phase retrieval
--------------------------

.. Todo:: Explain how phase retrieval is applied. 
