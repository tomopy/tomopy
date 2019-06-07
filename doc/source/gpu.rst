============
GPU Support
============

This section covers building TomoPy with support for GPU offloading and general usage. For the
implemented iterative algorithms, reconstruction times per-slice can be reduced by several
orders of magnitude depending on the hardware available. On a NVIDIA Volta (VX-100) GPU at NERSC, the
per-slice reconstruction time of the SIRT and MLEM algorithms for a 2048p image with 1501 projection
angles and 100 iterations reduced from ~6.5 hours to ~40 seconds.


Supported GPUs
--------------
TomoPy supports offloading to NVIDIA GPUs through compiled CUDA kernels on Linux and Windows 10.
NVIDIA GPUs on macOS are untested but likely supported.


Building TomoPy with CUDA
-------------------------
CMake is configured to automatically enable building GPU support when CMake can detect a valid CUDA compiler.
TomoPy requires CMake 3.9+, which has support for CUDA as a first-class language -- meaning that
the CUDA compiler only needs to be in the ``PATH``. On Unix, this is easily checked with the
command: ``which nvcc``. If the command returns a path to the compiler, build TomoPy normally.
If not, locate the CUDA compiler and place the path to the compiler in ``PATH``, remove the
build directory (``rm -r _skbuild`` or ``python setup.py clean``) and rebuild.

TomoPy includes the `Parallel Tasking Library (PTL) <https://github.com/jrmadsen/PTL>`_ as a git submodule
to handle the creation of a secondary thread-pool that assists in hiding the communication latency between
the CPU and GPU. This submodule is automatically checked out and compiled by the CMake build system.

Reconstructing with GPU offloading
----------------------------------

In order to reconstruct efficiently on the GPU, the algorithm has been implemented as a rotation-based
reconstruction instead of the standard ray-based reconstruction. The primary implication of the algorithmic
change is that when there are important pixels at the corners of the image, it will be necessary to pad the
image before reconstruction. This is due to the side-effect of a rotating at an arbitrary angle that is not
a factor of 90 degrees:

.. code-block:: python

        obj = tomopy.shepp2d()
        obj = tomopy.misc.morph.pad(obj, axis=1, mode='constant')
        obj = tomopy.misc.morph.pad(obj, axis=2, mode='constant')


Currently, the supported algorithms for GPU offloading are:

+--------------+
| Algorithms   |
+==============+
|  SIRT        |
+--------------+
|  MLEM        |
+--------------+

When GPU support is not available, due to either lack of compiler support or no CUDA devices available,
the algorithms will execute on the CPU with the same algorithm as the GPU version using OpenCV. When an
NVIDIA device is targeted, the algorithms utilize the NPP (NVIDIA Performance Primitives) library instead
of OpenCV which has limited GPU support. However, it is possible that GPU offloading will still occur
if OpenCV is configured with GPU support.

The addition of ``accelerated=True`` to ``tomopy.recon(...)`` is the only requirement for enabling
the accelerated versions of the above algorithms. However, there is support an additional customization:

=========================== ========= ===================================== ===========
``tomopy.recon`` parameters Type      Description                           Options
=========================== ========= ===================================== ===========
``accelerated``             boolean   Enable accelerated algorithm          True, False
--------------------------- --------- ------------------------------------- -----------
``pool_size``               int       Size of the secondary thread-pool
--------------------------- --------- ------------------------------------- -----------
``interpolation``           string    Interpolation scheme                  "NN", "LINEAR", "CUBIC"
--------------------------- --------- ------------------------------------- -----------
``device``                  string    Targeted device                       "cpu", "gpu"
--------------------------- --------- ------------------------------------- -----------
``grid_size``               nparray   GPU grid dimensions                   Set to ``[0,0,0]`` for auto grid size
--------------------------- --------- ------------------------------------- -----------
``block_size``              nparray   GPU block dimensions                  Default is ``[32,32,1]``
=========================== ========= ===================================== ===========

Multithreading
--------------
TomoPy supports multithreading at the Python level through the ``ncore`` parameter. When offloading to
the GPU, it is generally recommended to set ``ncore`` to the number of GPUs. As the threads started at the
Python level drop down into the compiled code of TomoPy, these threads increment a counter that spreads
their execution across all of the available GPUs

.. code-block:: cpp

    // thread counter for device assignment
    static std::atomic<int> ntid;

    // increment counter and get a "Python thread-id"
    int pythread_num = ntid++;

    // set the device to the modulus of number of device available
    int device       = pythread_num % cuda_device_count();

As mentioned previously, TomoPy creates a secondary thread-pool in the accelerated algorithms that assists
in hiding the communication latency between the CPU and GPU. Once a thread has been assigned a device,
it creates ``pool_size`` additional threads for this purpose. When offloading to the GPU, the standard
recommendation is to over-subscribe the number of threads relative to the number of hardware cores. The ideal
number of threads per GPU is around 12-24 threads. The default number of ``pool_size`` threads is twice the
number of hardware threads available divided by the number of threads started at the Python level, e.g. if
there are 8 CPU cores and 1 thread started at the Python level, 16 threads will be created in the secondary
thread-pool.
