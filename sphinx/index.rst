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
`Enthought Canopy <https://www.enthought.com/products/canopy/>`_. 
Enthought Canopy provides a robust Python framework with easy 
installation and has both free and commercial versions. 

Development
===========

TomoPy uses the `numpy/scipy documentation standard 
<https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_
for code development and `Sphinx <http://sphinx-doc.org/>`_ for
generating the documentation. Sphinx translates
`reST <http://docutils.sourceforge.net/rst.html>`_
formatted documents into html documents. All the documentation
files for TomoPy are in reST format. The reST documents can be
found under the ``sphinx/`` folder.

You need the have `Sphinx <http://sphinx-doc.org/>`_ to build the
documentation. It comes with the `Enthought Canopy
<http://www.enthought.com/products/canopy/>`_. To generate an HTML
documentation use::

    sphinx-build -b html sphinx/ doc

This creates a top-level build directory and puts the HTML
documentation inside that folder. Once the build is complete, 
the HTML documentation files are located under the ``doc/`` 
directory. You can use a browser to open the ``doc/index.html`` file.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
