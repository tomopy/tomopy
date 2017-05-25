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

    conda install -c dgursoy tomopy


Updating the installation
=========================

TomoPy is an active project, so we suggest you update your installation
frequently. To update the installation run::

    conda update -c dgursoy tomopy

For some more information about using Conda, please refer to the
`docs <http://conda.pydata.org/docs>`__.


Installing from source
======================

Sometimes an adventurous user may want to get the source code, which is 
always more up-to-date than the one provided by Conda 
(with more bugs of course!).

For this you need to get the source from the
`TomoPy repository <https://github.com/tomopy/tomopy>`_ on GitHub. 
You can get it to your local computer by opening a terminal and running::

    git clone https://github.com/tomopy/tomopy.git

Then in a terminal you need to run::

    python setup.py install

Keep in mind that you may need to install all dependencies listed in 
``requirements.txt`` or ``meta.yaml`` files manually or
as you please.


Importing TomoPy
================

When importing, it is best to import TomoPy before importing numpy.
See `this thread <https://github.com/tomopy/tomopy/issues/178>`_ for details.
