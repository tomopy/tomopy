About TomoPy
############

TomoPy is a Python toolbox to perform tomographic data processing and image reconstruction 
tasks at the Advanced Photon Source (http://www.aps.anl.gov). It uses the
HDF5 file format **Data Exchange** (https://github.com/data-exchange/data-exchange)
as the standard means of data exchange.

External Dependencies
*********************

==========  =======  ====================  ========================================================
dependency  version  URL                   comments
==========  =======  ====================  ========================================================
FFTW3       3.3.3    http://www.fftw.org   only float library is required
Boost C++   1.55.0   http://www.boost.org  only thread, system and date_time libraries are required
==========  =======  ====================  ========================================================

.. note:: For a quick install of these external dependencies, use::

       python install.py --boost --fftw
 
   provided with the TomoPy source code.  
   By default it installs them into ``/usr/local``. 
   If you wish to install them into a different directory, use::

       python install.py <desired-directory> --boost --fftw

   In this case, be sure that <desired-directory> is added to your
   **PYTHONPATH** environment variable for runtime use of TomoPy.

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
==========  =======  =====================================

Installing TomoPy
*****************

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
source      #. download latest source tarball from https://github.com/tomopy/tomopy/releases
            #. ``python setup.py install`` in the directory where *setup.py* resides
==========  ==========================================================================================
            

For some configurations you may need to specifically add 
the install directory to your ``PYTHONPATH``. To test if installation 
was succesfull, open a new command shell and try::

    python -c "import tomopy"

If it doesn't complain you are good to go!

