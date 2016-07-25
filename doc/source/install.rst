==================
Install directions
==================

This section covers the basics of how to download and install TomoPy.

.. contents:: Contents:
   :local:

Installing from source via command line
======================================
  
If you want to contribute to the development of TomoPy, you will 
need a local copy of the source code.  Clone from the 
`TomoPy repository <https://github.com/tomopy/tomopy>`_ on GitHub_::

    git clone https://github.com/tomopy/tomopy.git

then, inside your new cloned repo, run::

    python setup.py build
    sudo python setup.py install

Installing from Conda
=====================

If you only want to run TomoPy, not develop it, then 
you should install through a package manager.

First you must have `Conda <http://continuum.io/downloads>`_ installed, 
then open a terminal or a command prompt window and run::

    conda install -c dgursoy tomopy


Updating the installation
=========================

TomoPy is an active project, so we suggest you update your installation 
frequently. To update the installation run::

    conda update -c dgursoy tomopy

For some more information about using Conda, please refer to the 
`docs <http://conda.pydata.org/docs>`__.

Importing TomoPy
================

When importing, it is best to import TomoPy before importing numpy.  
See `this thread <https://github.com/tomopy/tomopy/issues/178>`_ for details.
