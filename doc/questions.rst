==========================
Frequently Asked Questions
==========================

:Page Status: Incomplete
:Last Reviewed: 2015-03-29


Here's a list of questions.

.. contents:: Questions
   :local:
   

How can I ask for help?
=======================

The easiest way to get help is to open an issue on GitHub. Simply:

1. Go to the `project GitHub page <https://github.com/dgursoy/tomopy>`_.

2. Click on `Issues <https://github.com/dgursoy/tomopy/issues>`_ 
   in the right menu tab.

3. Click on `New Issue <https://github.com/dgursoy/tomopy/issues/new>`_ 
   and write/submit your question.


How can I install tifffile plugin for scikit-image?
===================================================

The tifffile plugin is required to read and write 
three-dimensional TIFF stacks. Here's a guide on how to 
install it:

1. Download the file from 
   `here <http://www.lfd.uci.edu/~gohlke/code/tifffile.py>`_.

2. Find the scikit-image plugins directory (i.e., skimage/io/_plugins).
   
3. Move the downloaded file into the scikit-image plugins directory.

Now you should be able to use the plugin. To test it::

    from skimage import io as sio
    sio.use_plugin('tifffile', 'imsave')
   

Which papers should I cite when I use TomoPy?
===============================================

We kindly request you cite the following papers when you use TomoPy:

.. [#] Gursoy D, De Carlo F, Xiao X, Jacobsen C.
   TomoPy: A framework for the analysis of synchrotron tomographic data. 
   **Journal of Synchrotron Radiation**, 21(5):1188--1193, 2014. `[link] <http://dx.doi.org/10.1107/S1600577514013939>`__

.. [#] De Carlo F, Gursoy D, Marone F, Rivers M, Parkinson YD, Khan F, Schwarz N, Vine DJ, Vogt S, Gleber SC, Narayanan S, Newville M, Lanzirotti T, Sun Y, Hong YP, Jacobsen C.
    Scientific Data Exchange: a schema for HDF5-based storage of raw and analyzed data. 
    **Journal of Synchrotron Radiation**, 21(6):1224--1230, 2014. `[link] <http://dx.doi.org/10.1107/S160057751401604X>`__
    
