TomoPy 1.0.0 Release Notes
==========================

.. contents:: 
   :local:


New features
------------

* `FFTW <www.fftw.org>`_ implementation is now adopted. All functions that rely on FFTs such as gridrec, phase retrieval, stripe removal, etc. are now using the FFTW implementation through `PyFFTW <https://hgomersall.github.io/pyFFTW/>`_. 

* ``sinogram_order`` is added to ``recon`` as an additional argument. It determines whether data is a stack of sinograms (True, y-axis first axis) or a stack of radiographs (False). Default is False, but we plan to make it True in the upcoming release.

* Reconstruction algorithms only copies data if necessary.

* Updated library to support new mproc and recon functions. The data is now passed in sinogram order to recon functions. Also updated tests.

* ``ncores`` and ``nchunks`` are now independent.

* Setting ``nchunks`` to zero removes the dimension. That allows for the functions work on 2D data rather than 3D data.

* Sliced data are used so that each process only receives the data it needs. No more ``istart`` and ``iend`` variables for setting up indices in parallel processes.
    
* Functions will reuse sharedmem arrays if they can.


New functions
-------------

* `minus_log <http://tomopy.readthedocs.io/en/latest/api/tomopy.prep.normalize.html#tomopy.prep.normalize.minus_log>`_
* `trim_sinogram <http://tomopy.readthedocs.org/en/latest/api/tomopy.misc.morph.html#tomopy.misc.morph.trim_sinogram>`_


New packages in Conda channel
-----------------------------

* `dxchange 0.1.1 <https://anaconda.org/dgursoy/dxchange>`_
* `fftw 3.3.4 <https://anaconda.org/dgursoy/fftw>`_
* `pyfftw 0.9.2 <https://anaconda.org/dgursoy/pyfftw>`_
* `pywavelets 0.4.0 <https://anaconda.org/dgursoy/pywavelets>`_
* `xraylib 3.1.0 <https://anaconda.org/dgursoy/xraylib>`_


Deprecated features
-------------------

*  All data I/O related functions are deprecated. They are available through `DXchange <http://dxchange.rtfd.org>`_ package.

* Removed fft.h and fft.c, they are now completely replaced with FFTW.


Backward incompatible changes
-----------------------------

* ``emission`` argument is removed from ``recon``. After this change the tomographic image reconstruction algorithms always assume data to be normalized.


Contributors
------------

* Arthur Glowacki (`@aglowacki`_)
* Daniel Pelt (`@dmpelt`_)
* Doga Gursoy (`@dgursoy`_)
* Francesco De Carlo (`@decarlof`_)
* Lin Jiao (`@yxqd`_)
* Luis Barroso-Luque (`@lbluque`_)
* Michael Sutherland (`@michael-sutherland`_)
* Rafael Vescovi (`@ravescovi`_)
* Thomas Caswell (`@tacaswell`_)
* Pete R. Jemian (`@prjemian`_)

.. _`@aglowacki`: https://github.com/aglowacki
.. _`@dmpelt`: https://github.com/dmpelt
.. _`@dgursoy`: https://github.com/dgursoy
.. _`@dakefeng`: https://github.com/dakefeng
.. _`@decarlof`: https://github.com/decarlof
.. _`@lbluque`: https://github.com/lbluque
.. _`@yxqd`: https://github.com/yxqd
.. _`@michael-sutherland`: https://github.com/michael-sutherland
.. _`@ravescovi`: https://github.com/ravescovi
.. _`@tacaswell`: https://github.com/tacaswell
.. _`@prjemian`: https://github.com/prjemian
.. _`@celiafish`: https://github.com/celiafish
