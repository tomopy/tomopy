#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

extern "C"
{
#include "prep.h"
}

namespace py = pybind11;

using farray = py::array_t<float, py::array::c_style | py::array::forcecast>;

PYBIND11_MODULE(normalize, m)
{
    m.doc() = "Module for data normalization";

    m.def(
        "normalize_bg",
        [](farray tomo, int air, int ncore, int nchunk)
        {
            // nchunk ignored because OpenMP is shared memory
            py::gil_scoped_release release;
            auto buf1 = tomo.mutable_unchecked<3>();
            normalize_bg(
                buf1.mutable_data(0, 0, 0), buf1.shape(0), buf1.shape(1), buf1.shape(2),
                air, ncore
            );
            return tomo;
        },
R"(
Normalize 3D tomgraphy data based on background intensity.

Weight sinogram such that the left and right image boundaries
(i.e., typically the air region around the object) are set to one
and all intermediate values are scaled linearly.

Parameters
----------
tomo : ndarray
    3D tomographic data.
air : int, optional
    Number of pixels at each boundary to calculate the scaling factor.
ncore : int, optional
    Number of cores that will be assigned to jobs.
nchunk : int, optional
    This parameter is ignored because OpenMP is shared memory.

Returns
-------
ndarray
    Corrected 3D tomographic data.
)",
        py::arg("tomo"),
        py::arg("air") = 1,
        py::arg("ncore") = 1, // TODO: Must default None to not break API
        py::arg("nchunk") = 0
    );
}
