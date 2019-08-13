
TomoPy with the ASTRA toolbox
--------------------------------------

Here is an example on how to use the `ASTRA toolbox
<http://www.astra-toolbox.com/docs/install.html>`__ through its integration with
`TomoPy <http://tomopy.readthedocs.io/en/latest/>`__, as published in
:cite:`Pelt:16a`.

To reconstruct the image with the ASTRA toolbox instead of TomoPy, change the
``algorithm`` keyword to ``tomopy.astra``. Specify which projection kernel to
use (``proj_type``) and which ASTRA algorithm to reconstruct with (``method``)
in the ``options`` keyword.

For example, to use a line-based CPU kernel and the FBP method, use:

.. code:: python

    options = {'proj_type': 'linear', 'method': 'FBP'}
    recon = tomopy.recon(proj,
                         theta,
                         center=rot_center,
                         algorithm=tomopy.astra,
                         options=options)
    recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)
    plt.imshow(recon[0, :, :], cmap='Greys_r')
    plt.show()

.. image:: astra_files/output_30_0.png

If you have a CUDA-capable NVIDIA GPU, reconstruction times can be greatly
reduced by using GPU-based algorithms of the ASTRA toolbox, especially for
iterative reconstruction methods.

To use the GPU, change the ``proj_type`` option to ``'cuda'``, and use
CUDA-specific algorithms (e.g. ``'FBP_CUDA'`` for FBP):

.. code:: python

    options = {'proj_type': 'cuda', 'method': 'FBP_CUDA'}
    recon = tomopy.recon(proj,
                         theta,
                         center=rot_center,
                         algorithm=tomopy.astra,
                         options=options)
    recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)
    plt.imshow(recon[0, :, :], cmap='Greys_r')
    plt.show()

.. image:: astra_files/output_32_0.png

Many algorithms of the ASTRA toolbox support additional options, which can be
found in the `documentation
<http://www.astra-toolbox.com/docs/algs/index.html>`__. These options can be
specified using the ``extra_options`` keyword.

For example, to use the GPU-based iterative SIRT method with a
non-negativity constraint, use:

.. code:: python

    extra_options = {'MinConstraint': 0}
    options = {
        'proj_type': 'cuda',
        'method': 'SIRT_CUDA',
        'num_iter': 200,
        'extra_options': extra_options
    }
    recon = tomopy.recon(proj,
                         theta,
                         center=rot_center,
                         algorithm=tomopy.astra,
                         options=options)
    recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)
    plt.imshow(recon[0, :, :], cmap='Greys_r')
    plt.show()


.. image:: astra_files/output_34_0.png

More information about the projection kernels and algorithms that are supported
by the ASTRA toolbox can be found in the documentation: `projection kernels
<http://www.astra-toolbox.com/docs/proj2d.html>`__ and `algorithms
<http://www.astra-toolbox.com/docs/algs/index.html>`__. Note that only the 2D
(i.e. slice-based) algorithms are supported when reconstructing through TomoPy.
