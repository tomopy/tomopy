TomoPy
######

.. image:: https://readthedocs.org/projects/tomopy/badge/?version=latest
   :target: https://readthedocs.org/projects/tomopy/?badge=latest
   :alt: Read the Docs

.. image:: https://travis-ci.org/tomopy/tomopy.svg?branch=master
   :target: https://travis-ci.org/tomopy/tomopy
   :alt: Travis CI

.. image:: https://coveralls.io/repos/dgursoy/tomopy/badge.svg?branch=master 
   :target: https://coveralls.io/r/tomopy/tomopy?branch=master
   :alt: Coveralls
   
.. image:: https://codeclimate.com/github/tomopy/tomopy/badges/gpa.svg
   :target: https://codeclimate.com/github/tomopy/tomopy
   :alt: Code Climate

.. image:: https://anaconda.org/dgursoy/tomopy/badges/downloads.svg
   :target: https://anaconda.org/dgursoy/tomopy
   :alt: Anaconda downloads

**TomoPy** is an open-source Python package for tomographic data 
processing and image reconstruction.

Features
========

* Image reconstruction algorithms for tomography.
* Various filters, ring removal algorithms, phase retrieval algorithms.
* Forward projection operator for absorption and wave propagation.

Installation
============

Have `Conda <http://continuum.io/downloads>`_ installed first,  
then open a terminal or a command prompt window and run:

    conda install -c dgursoy tomopy
    
# Windows 10 

Make sure your anaconda version > 4.4.0 and you are using mingw compiler

Install two required packages from conda first

    pip install pyffwt
    conda install -c conda-forge pthreads-win32=2.9.1
    conda install -c salilab fftw=3.34
    conda install -c clinicalgraphics pywt=0.3.0 

Setup your envirnment for fftw libraries

    LD_LIBRARY_PATH=[ANACONDA_ROOT]\\Library\\lib
    C_INCLUDE_PATH=[ANACONDA_ROOT]\\Library\\include

Then install tomopy with pip

    pip install .


Contribute
==========

* Issue Tracker: https://github.com/tomopy/tomopy/issues
* Documentation: https://github.com/tomopy/tomopy/tree/master/doc
* Source Code: https://github.com/tomopy/tomopy/tree/master/tomopy
* Tests: https://github.com/tomopy/tomopy/tree/master/test

License
=======

The project is licensed under the 
`BSD-3 <https://github.com/tomopy/tomopy/blob/master/LICENSE.txt>`_ license.
