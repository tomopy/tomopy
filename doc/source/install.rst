==================
Install directions
==================

This section covers the basics of how to download and install TomoPy.

.. contents:: Contents:
   :local:

Installing from Conda (Recommended)
===================================

If you only want to run TomoPy, not develop it, then you should install through
a package manager.  The package manager Conda can install Tomopy and its
dependencies for you.

First you must have `Conda <http://continuum.io/downloads>`_ installed,
then open a terminal or a command prompt window and run::

    conda install -c conda-forge tomopy


Updating the installation

TomoPy is an active project, so we suggest you update your installation
frequently. To update the installation run::

    conda update -c conda-forge tomopy

For some more information about using Conda, please refer to the
`docs <http://conda.pydata.org/docs>`__.


Installing from source

Importing TomoPy
================

When importing, it is best to import TomoPy before importing numpy.
See `this thread <https://github.com/tomopy/tomopy/issues/178>`_ for details.
