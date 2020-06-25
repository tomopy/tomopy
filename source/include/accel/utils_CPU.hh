//  Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.
//  Copyright 2015. UChicago Argonne, LLC. This software was produced
//  under U.S. Government contract DE-AC02-06CH11357 for Argonne National
//  Laboratory (ANL), which is operated by UChicago Argonne, LLC for the
//  U.S. Department of Energy. The U.S. Government has rights to use,
//  reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR
//  UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
//  ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is
//  modified to produce derivative works, such modified software should
//  be clearly marked, so as not to confuse it with the version available
//  from ANL.
//  Additionally, redistribution and use in source and binary forms, with
//  or without modification, are permitted provided that the following
//  conditions are met:
//      * Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//      * Redistributions in binary form must reproduce the above copyright
//        notice, this list of conditions and the following disclaimer in
//        the documentation andwith the
//        distribution.
//      * Neither the name of UChicago Argonne, LLC, Argonne National
//        Laboratory, ANL, the U.S. Government, nor the names of its
//        contributors may be used to endorse or promote products derived
//        from this software without specific prior written permission.
//  THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
//  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago
//  Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
//  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
//  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
//  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//  ---------------------------------------------------------------
//   TOMOPY header

#pragma once

//--------------------------------------------------------------------------------------//

#include "common_CPU.hh"
#include "macros.hh"
#include "utils_CPU.hh"

BEGIN_EXTERN_C
#include "cxx_extern.h"
END_EXTERN_C

//--------------------------------------------------------------------------------------//
#if defined(WIN32)
#    define _Complex
#endif

#define CPU_NN CV_INTER_NN
#define CPU_LINEAR CV_INTER_LINEAR
#define CPU_AREA CV_INTER_AREA
#define CPU_CUBIC CV_INTER_CUBIC
#define CPU_LANCZOS CV_INTER_LANCZOS4

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct OpenCVDataType
{
    template <typename _Up = _Tp>
    static constexpr int value()
    {
        static_assert(std::is_same<_Up, _Tp>::value, "OpenCV data type not overloaded");
        return -1;
    }
};

#define DEFINE_OPENCV_DATA_TYPE(pod_type, opencv_type)                                   \
    template <>                                                                          \
    struct OpenCVDataType<pod_type>                                                      \
    {                                                                                    \
        template <typename _Up = pod_type>                                               \
        static constexpr int value()                                                     \
        {                                                                                \
            return opencv_type;                                                          \
        }                                                                                \
    };

// floating point types
DEFINE_OPENCV_DATA_TYPE(float, CV_32F)
DEFINE_OPENCV_DATA_TYPE(double, CV_64F)

// signed integer types
DEFINE_OPENCV_DATA_TYPE(int8_t, CV_8S)
DEFINE_OPENCV_DATA_TYPE(int16_t, CV_16S)
DEFINE_OPENCV_DATA_TYPE(int32_t, CV_32S)

// unsigned integer types
DEFINE_OPENCV_DATA_TYPE(uint8_t, CV_8U)
DEFINE_OPENCV_DATA_TYPE(uint16_t, CV_16U)

#undef DEFINE_OPENCV_DATA_TYPE  // don't pollute

//--------------------------------------------------------------------------------------//

inline int
GetOpenCVInterpolationMode(const std::string& preferred)
{
    EnvChoiceList<int> choices = {
        EnvChoice<int>(CPU_NN, "NN", "nearest neighbor interpolation"),
        EnvChoice<int>(CPU_LINEAR, "LINEAR", "bilinear interpolation"),
        EnvChoice<int>(CPU_CUBIC, "CUBIC", "bicubic interpolation")
    };
    return GetChoice<int>(choices, preferred);
}

//--------------------------------------------------------------------------------------//

inline cv::Mat
opencv_affine_transform(const cv::Mat& warp_src, double theta, const int& nx,
                        const int& ny, int eInterp, double scale = 1.0)
{
    cv::Mat   warp_dst = cv::Mat::zeros(nx, ny, warp_src.type());
    double    cx       = 0.5 * ny + ((ny % 2 == 0) ? 0.5 : 0.0);
    double    cy       = 0.5 * nx + ((nx % 2 == 0) ? 0.5 : 0.0);
    cv::Point center   = cv::Point(cx, cy);
    cv::Mat   rot      = cv::getRotationMatrix2D(center, theta, scale);
    cv::warpAffine(warp_src, warp_dst, rot, warp_src.size(), eInterp);
    return warp_dst;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void
cxx_rotate_ip(array_t<_Tp>& dst, const _Tp* src, double theta, const int& nx,
              const int& ny, int eInterp, double scale = 1.0)
{
    cv::Mat warp_src = cv::Mat::zeros(nx, ny, OpenCVDataType<_Tp>::value());
    memcpy(warp_src.ptr(), src, nx * ny * sizeof(float));
    cv::Mat warp_rot =
        opencv_affine_transform(warp_src, theta * degrees, nx, ny, eInterp, scale);
    memcpy(dst.data(), warp_rot.ptr(), nx * ny * sizeof(float));
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
array_t<_Tp>
cxx_rotate(const _Tp* src, double theta, const intmax_t& nx, const intmax_t& ny,
           int eInterp, double scale = 1.0)
{
    array_t<_Tp> dst(nx * ny, _Tp());
    cxx_rotate_ip(dst, src, theta, nx, ny, eInterp, scale);
    return dst;
}

//--------------------------------------------------------------------------------------//

inline iarray_t
cxx_compute_sum_dist(int dy, int dt, int dx, int nx, int ny, const float* theta)
{
    auto compute = [&](const iarray_t& ones, iarray_t& sum_dist, int p) {
        for(int s = 0; s < dy; ++s)
        {
            for(int d = 0; d < dx; ++d)
            {
                int32_t*       _sum_dist = sum_dist.data() + (s * nx * ny) + (d * nx);
                const int32_t* _ones     = ones.data() + (d * nx);
                for(int n = 0; n < nx; ++n)
                {
                    _sum_dist[n] += (_ones[n] > 0) ? 1 : 0;
                }
            }
        }
    };

    iarray_t rot(nx * ny, 0);
    iarray_t tmp(nx * ny, 1);
    iarray_t sum_dist(dy * nx * ny, 0);

    for(int p = 0; p < dt; ++p)
    {
        float theta_p_rad = fmodf(theta[p] + halfpi, twopi);
        cxx_rotate_ip(rot, tmp.data(), -theta_p_rad, nx, ny, CPU_NN);
        compute(rot, sum_dist, p);
    }

    return sum_dist;
}
