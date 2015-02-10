.. APS Imaging toolbox

.. _installation:

=================
Installing TomoPy
=================

Make sure you have Python 2.6 or 2.7
and these dependencies installed in your system. 

======  ==============================================
Python  URL
======  ==============================================
2.6     http://www.python.org/download/releases/2.6
2.7     http://www.python.org/download/releases/2.7
======  ==============================================

.. note:: This code was developed using the Enthought Python
   Distribution (www.enthought.com) which satisfies most of the
   above-stated Python Dependencies.

External Dependencies
*********************

========== =======  ====================  ========================================================
dependency version  URL 		  comments
========== =======  ====================  ========================================================
FFTW3	   3.3.3    http://www.fftw.org   only float library is required
Boost C++  1.55.0   http://www.boost.org  only thread, system and date_time libraries are required
========== =======  ====================  ========================================================

.. Data Exchange devel    https://github.com/data-exchange/data-exchange  separate project in parallel development

Quick Install
-------------

For a quick install of these external dependencies, use::

       python install.py --boost --fftw
 
provided with the TomoPy source code.  
By default it installs them into ``/usr/local``. 
If you wish to install them into a different directory, use::

       python install.py <desired-directory> --boost --fftw

In this case, be sure that <desired-directory> is added to your
environment variables for runtime use of TomoPy::
   
       setenv LD_LIBRARY_PATH <desired-directory>/lib
       setenv C_INCLUDE_PATH <desired-directory>/include

Python Dependencies
*******************

==========  =======  =====================================
dependency  version  URL
==========  =======  =====================================
NumPy       1.8.0    http://www.numpy.org
SciPy       0.13.2   http://www.scipy.org
H5Py        2.2.1    http://www.h5py.org
PyWt        0.2.2    http://www.pybytes.com/pywavelets
Pillow      2.3.0    https://pypi.python.org/pypi/Pillow
Skimage     0.10     http://scikit-image.org
==========  =======  =====================================

Environment Variables
*********************

Make sure you have Python 2.6 or 2.7
and the above dependencies installed in your system. 

======  ==============================================
Python  URL
======  ==============================================
2.6     http://www.python.org/download/releases/2.6
2.7     http://www.python.org/download/releases/2.7
======  ==============================================

.. note:: This code was developed using the Enthought Python
   Distribution (www.enthought.com) which satisfies most of the
   above-stated Python Dependencies.

Then, permanently set the following environment variables in the shell.
For */bin/csh* or */bin/tcsh* (note: this will overwrite any previous 
definitions of these terms)::

    setenv LD_LIBRARY_PATH <my-boost-lib-dir>:<my-fftw-lib-dir>
    setenv C_INCLUDE_PATH <my-boost-include-dir>:<my-fftw-include-dir>
    
For */bin/bash*::

    export LD_LIBRARY_PATH=<my-boost-lib-dir>:<my-fftw-lib-dir>
    export C_INCLUDE_PATH=<my-boost-include-dir>:<my-fftw-include-dir>

Once the environment variables are set correctly, then any of these methods should work:

==========  ==========================================================================================
from        procedure
==========  ==========================================================================================
Python egg  #. download latest released egg from https://github.com/tomopy/tomopy/releases
            #. ``easy_install my-egg-name`` in the directory where the egg resides.
release     #. download latest source tarball from https://github.com/tomopy/tomopy/releases
            #. ``python setup.py install`` in the directory where *setup.py* resides
github      #. clone the tomopy repository ``git clone https://github.com/tomopy/tomopy.git tomopy``
            #. ``python setup.py install`` in the ``tomopy`` directory where *setup.py* resides
==========  ==========================================================================================
            
For some configurations you may need to specifically add 
the install directory to your ``PYTHONPATH``. 

To test if installation was succesfull:

#. open a new command shell
#. **change to a different directory than the tomopy source**
#. try::

    python -c "import tomopy"

If it doesn't complain you are good to go!

Install Example
---------------

Here is a complete example of the installation, including
a different *<desired-directory>* for Boost, fftw3, and tomopy:

#. download latest source .tar.gz from https://github.com/tomopy/tomopy/releases
#. expand the source .tar.gz into a new directory, build and install with these commands::

     /bin/tcsh
     setenv SANDBOX /tmp/sandbox
     mkdir -p $SANDBOX/lib/python2.7/site-packages/
     cd /tmp
     tar xzf ~/Downloads/tomopy-0.0.2.tar.gz
     cd tomopy-0.0.2/
     python install.py $SANDBOX --boost --fftw
     setenv LD_LIBRARY_PATH $SANDBOX/lib
     setenv C_INCLUDE_PATH $SANDBOX/include
     setenv PYTHONPATH $SANDBOX/lib/python2.7/site-packages/
     python setup.py install --prefix=$SANDBOX
     cd /tmp
 
     echo "SANDBOX = $SANDBOX"
     echo "LD_LIBRARY_PATH = $LD_LIBRARY_PATH"
     echo "C_INCLUDE_PATH = $C_INCLUDE_PATH"
     echo "PYTHONPATH = $PYTHONPATH"
     python -c "import tomopy"

Windows Installation
********************
#. Download and install Anaconda Windows 64-bit Python 2.7 (the install has options for non-admin installs):
     http://continuum.io/downloads
#. Open command prompt and run:
     ``pip install -U pywavelets``
#. Download FFTW3 for windows:
     ftp://ftp.fftw.org/pub/fftw/fftw-3.3.4-dll64.zip
#. Unzip and copy libfftw3f-3.dll to anaconda root directory, ex. C:\\Anaconda\
#. Download tomopy.egg from the build server: 
     https://jenkins.aps.anl.gov/view/Python/job/Tomopy_trunk/ws/dist/tomopy-0.0.3-py2.7-win-amd64.egg
#. Open command prompt where tomopy.egg is saved and run:
     ``easy_install tomopy-0.0.3-py2.7-win-amd64.egg``
