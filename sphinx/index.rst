.. APS Imaging toolbox
   sphinx-quickstart on Thu Oct 24 17:20:31 2013.

Welcome to TomoPy's documentation!
*********************************

About TomoPy
============

TomoPy is a Python toolbox to perform data processing and
image reconstruction tasks at APS. It uses `HDF5 file format
<https://subversion.xray.aps.anl.gov/DataExchange/doc/trunk/>`_ as
the standard means of data exchange.

Installing TomoPy
=================

The easiest way to install the required packages is to download the
`Enthought Python Distribution (EPD)
<http://www.enthought.com/products/epd/>`_. EPD has both free and
commercial versions and include all the required packages used in
the toolbox.

.. note::
    APS users can use the EPD which is already installed under
    ``/APSshare`` by adding the following line to the .cshrc file
    located in their home directory::

        alias python /APSshare/epd/rh6-x86_64/bin/python

Development
===========

TomoPy uses the `numpy/scipy documentation standard <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_
for code development and `Sphinx <http://sphinx-doc.org/>`_ for
generating the documentation. Sphinx translates
`reST <http://docutils.sourceforge.net/rst.html>`_
formatted documents into html documents. All the documentation
files for TomoPy are in reST format. The reST documents can be
found under the ``sphinx/`` folder.

You need the have `Sphinx <http://sphinx-doc.org/>`_ to build the
documentation. It comes with the `Enthought Python Distribution
<http://www.enthought.com/products/epd/>`_. To generate an HTML
documentation use::

    sphinx-build -b html sphinx/ doc

This creates a top-level build directory and puts the HTML
documentation inside that folder. For more options you can use::

    sphinx-build -help

.. note::
    APS users can use the Sphinx which is already installed
    under ``/APSshare`` by adding the following line to the
    .cshrc file located in their home directory::

        alias sphinx-build /APSshare/epd/rh6-x86_64/bin/sphinx-build

Once the build is complete, the HTML documentation files are
located under the ``doc/`` directory. You can use a browser
to open the ``doc/index.html`` file.

Tutorials
=========

Reading data
------------

First import the ``data`` module in ``inout`` package::

    from inout import data

Then, we can import the HDF file by using ``read`` function::

    inputFile = 'mydata.hdf'
    mydata = data.read(inputFile)

    # Print projection data
    print mydata.data

    # Print white field data
    print mydata.white

    # Print dark field data
    print mydata.dark

When the dataset is huge, it is easier to import a subset
of the whole dataset (i. e., a few slices or a few projections only)
to try some initial parameters to find optimal reconstruction
parameters. To do so, one can use additional arguments as folows::

    mydata = data.read(inputFile, slicesStart=100, slicesEnd=200, slicesStep=2)
    print mydata.data.shape

This will import only a subset of the available slices, i. e. the
slices 100, 102, 104, ... , 198, 200. One can also undersample the
projections as::

    mydata = data.read(inputFile, projectionsStep=2)
    print mydata.data.shape

or select a subset of the available projections::

    mydata = data.read(inputFile, projectionsStart=100, projectionsEnd=200)
    print mydata.data.shape

Similarly, one can select a subset of the white or dark field data
for reading as::

    mydata1 = data.read(inputFile, whiteStart=3, whiteEnd=6)
    print mydata2.white.shape
    mydata2 = data.read(inputFile, darkStart=6, darkEnd=8)
    print mydata1.white.shape

This will use only the white field measurements 3, 4, 5 and 6 for
``mydata1`` and dark field measurements 6, 7 and 8 for ``mydata2``.
In the extreme one can combine all these options to carve the HDF file
and import as desired::

    mydata = data.read(inputFile, projectionsStart=100, projectionsEnd=200, projectionsStep=2, slicesStart=100, slicesEnd=200, slicesStep=2, pixelsStart=100, pixelsEnd=800, pixelsStep=4, whiteStart=2, whiteEnd=5, darkStart=10, darkEnd=14)


Normalizing data
----------------

Let us first assume we have imported the data using::

    from inout import data
    inputFile = 'mydata.hdf'
    mydata = data.read(inputFile)

To normalize data with the average white field measurements, simply
use::

    mydata.normalize()

It will replace the ``dataset.data`` with the normalized values.
It is overwritten because datasets are usually too large to
duplicate.

Filtering data
--------------

Generally median filtering is used to denoise data before the
reconstruction process. By default a 1-D median filter is applied on
the projections for each slice independently. For this use::

    mydata.medianFilter()

It overwrites the dataset with the filtered values. The size of
the filter is 3 pixels, but one can modify the size of the filter
by::

   mydata.medianFilter(size=(1, 4))

``size`` determines the size of the filter acting on
the projection image. The first argument is the x-axis (slices)
and the second argument refers to the y-axis (pixels). One can also
apply filtering on other planes with different sizes. For example
the following applies a 3 by 3 filter to each sinogram::

   mydata.medianFilter(axis=0, size=(3, 3))

``axis`` (either 0, 1 or 2) determines at the axis of the plane
for the median filter to be applied.

Finding rotation center
-----------------------
The location of the rotation center during tomographic data
acquisition can be found by::

     mydata.optimizeCenter()

This routine updates the rotation location (with half-pixel
accuracy) with the one that minimizes the entropy of the
reconstructed image of central slice. It uses ``Nelder-Mead``
method for optimization. Another example of the usage of this
function with different arguments is::

    mydata.optimizeCenter(sliceNo=100, inCenter=100, tol=1, fiterSigma=2)

``sliceNo`` refers to the slice index among the whole stack of
slices to be used for optimization (rather than using the central
slice). ``inCenter`` refers to the initial guess for the center
point. By default the central pixel is assumed to be the initial
guess. If this number is very different than the rotation center
the optimization may fail. ``tol`` represents the tolerance in
terms of pixel size. Smaller ``tol`` values takes longer times
to compute. ``filterSigma`` represents the variance of the 2-D
Gaussian filter in terms of pixels which is convolved with the
data before optimization. This value can be selected higher
particularly for images where the high-frequency terms are
dominating in the data (e.g., phase-contrast dataset).

Removal of stripes
------------------

Frodo's ring must be destroyed!


Phase retrieval
---------------

Phase images can be retrieved by::

    mydata.retrievePhase(pixelSize=1e-4, dist=100, energy=20)

The data is owerritten by the phase values. ``pixelSize`` is the
detector size in cm. ``dist`` is the sample to detector distance
in cm. ``energy`` is the incident x-ray energy in keV.

Tomographic reconstruction
--------------------------

First, initialize the reconstruction parameters::

    from recon import tomoRecon
    recon = tomoRecon.tomoRecon(mydata)

The outout object ``recon`` contains the reconstruction parameters.
Then, the reconstruction is calculated with these parameters::

    recon.run(mydata)

The reconstructed values are stored in ``recon.data``.

Software
========

.. toctree::
   :maxdepth: 2

   dataio
   tomoRecon
   visualize

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
