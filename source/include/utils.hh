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
//   TOMOPY class header

#pragma once

#include "common.hh"
#include "gpu.hh"

//--------------------------------------------------------------------------------------//

#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#if defined(TOMOPY_USE_OPENCV)
#    include <opencv2/highgui/highgui.hpp>
#    include <opencv2/imgcodecs.hpp>
#    include <opencv2/imgproc/imgproc.hpp>
#endif

#if defined(TOMOPY_USE_IPP)
#    include <ipp.h>
#    include <ippdefs.h>
#    include <ippi.h>
#endif

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void
print_gpu_array(const uintmax_t& n, const _Tp* gpu_data, const int& itr, const int& slice,
                const int& angle, const int& pixel, const std::string& tag)
{
    std::ofstream     ofs;
    std::stringstream fname;
    fname << "outputs/gpu/" << tag << "_" << itr << "_" << slice << "_" << angle << "_"
          << pixel << ".dat";
    ofs.open(fname.str().c_str());
    std::vector<_Tp> cpu_data(n, _Tp());
    std::cout << "printing to file " << fname.str() << "..." << std::endl;
    gpu2cpu_memcpy<_Tp>(gpu_data, cpu_data.data(), n, 0);
    if(!ofs)
        return;
    for(uintmax_t i = 0; i < n; ++i)
        ofs << std::setw(6) << i << " \t " << std::setw(12) << std::setprecision(8)
            << cpu_data[i] << std::endl;
    ofs.close();
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void
print_cpu_array(const uintmax_t& nx, const uintmax_t& ny, const _Tp* data, const int& itr,
                const int& slice, const int& angle, const int& pixel,
                const std::string& tag)
{
    std::ofstream     ofs;
    std::stringstream fname;
    fname << "outputs/cpu/" << tag << "_" << itr << "_" << slice << "_" << angle << "_"
          << pixel << ".dat";
    std::stringstream ss;
    for(uintmax_t j = 0; j < ny; ++j)
    {
        for(uintmax_t i = 0; i < nx; ++i)
        {
            ss << std::setw(6) << i << " \t " << std::setw(12) << std::setprecision(8)
               << data[i + j * nx] << std::endl;
        }
        ss << std::endl;
    }
    ofs.open(fname.str().c_str());
    if(!ofs)
        return;
    ofs << ss.str() << std::endl;
    ofs.close();
}

//======================================================================================//

template <typename _Tp>
void
ipp_affine_transform(array_t<_Tp>& dst, const _Tp* src, double theta_rad,
                     double theta_deg, const intmax_t& nx, const intmax_t& ny,
                     double scale = 1.0)
{
    std::stringstream ss;
    ss << __FUNCTION__ << " not implemented with Intel IPP!";
    throw std::runtime_error(ss.str());
}

//======================================================================================//

#if defined(TOMOPY_USE_OPENCV)

#    define CPU_NN CV_INTER_NN
#    define CPU_LINEAR CV_INTER_LINEAR
#    define CPU_AREA CV_INTER_AREA
#    define CPU_CUBIC CV_INTER_CUBIC
#    define CPU_LANCZOS CV_INTER_LANCZOS4

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

#    define DEFINE_OPENCV_DATA_TYPE(pod_type, opencv_type)                               \
        template <>                                                                      \
        struct OpenCVDataType<pod_type>                                                  \
        {                                                                                \
            template <typename _Up = pod_type>                                           \
            static constexpr int value()                                                 \
            {                                                                            \
                return opencv_type;                                                      \
            }                                                                            \
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

#    undef DEFINE_OPENCV_DATA_TYPE  // don't pollute

//--------------------------------------------------------------------------------------//

inline int
GetOpenCVInterpolationMode()
{
    static int eInterp =
        GetEnv<int>("TOMOPY_OPENCV_INTER", GetEnv<int>("TOMOPY_INTER", CPU_CUBIC));
    return eInterp;
}

//--------------------------------------------------------------------------------------//

inline cv::Mat
opencv_affine_transform(const cv::Mat& warp_src, double theta, const intmax_t& nx,
                        const intmax_t& ny, int eInterp = GetOpenCVInterpolationMode(),
                        double scale = 1.0)
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
cxx_rotate_ip(array_t<_Tp>& dst, const _Tp* src, double theta, const intmax_t& nx,
              const intmax_t& ny, int eInterp = GetOpenCVInterpolationMode(),
              double scale = 1.0)
{
    memset(dst.data(), 0, nx * ny * sizeof(_Tp));
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
           int eInterp = GetOpenCVInterpolationMode(), double scale = 1.0)
{
    array_t<_Tp> dst(nx * ny, _Tp());
    cxx_rotate_ip(dst, src, theta, nx, ny, eInterp, scale);
    return dst;
}

#else  // TOMOPY_USE_OPENCV

#    define CPU_NN 0
#    define CPU_LINEAR 1
#    define CPU_AREA 3
#    define CPU_CUBIC 4
#    define CPU_LANCZOS 5

//======================================================================================//

template <typename _Tp>
void
cxx_rotate_ip(array_t<_Tp>& dst, const _Tp* src, double theta, const intmax_t& nx,
              const intmax_t& ny, int = 0, double scale = 1.0)
{
    memset(dst.data(), 0, nx * ny * sizeof(_Tp));
#    if defined(TOMOPY_USE_IPP)
    try
    {
        ipp_affine_transform(dst, src, theta, theta * degrees, nx, ny, scale);
    }
    catch(std::exception& e)
#    endif
    {
        std::cerr << e.what() << "\n";
        // this is flawed and should not be production
        int   src_size = nx * ny;
        float xoff     = (0.5f * nx) - 0.5f;
        float yoff     = (0.5f * ny) - 0.5f;

        for(int j = 0; j < ny; ++j)
        {
            for(int i = 0; i < nx; ++i)
            {
                // indices in 2D
                float rx = float(i) - xoff;
                float ry = float(j) - yoff;
                // transformation
                float tx = rx * cosf(theta) + -ry * sinf(theta);
                float ty = rx * sinf(theta) + ry * cosf(theta);
                // indices in 2D
                float x = (tx + xoff);
                float y = (ty + yoff);
                // index in 1D array
                int  rz    = j * nx + i;
                auto index = [&](int _x, int _y) { return _y * nx + _x; };
                // within bounds
                int   x1    = floorf(tx + xoff);
                int   y1    = floorf(ty + yoff);
                int   x2    = x1 + 1;
                int   y2    = y1 + 1;
                float fxy1  = 0.0f;
                float fxy2  = 0.0f;
                int   ixy11 = index(x1, y1);
                int   ixy21 = index(x2, y1);
                int   ixy12 = index(x1, y2);
                int   ixy22 = index(x2, y2);
                if(ixy11 >= 0 && ixy11 < src_size)
                    fxy1 += (x2 - x) * src[ixy11];
                if(ixy21 >= 0 && ixy21 < src_size)
                    fxy1 += (x - x1) * src[ixy21];
                if(ixy12 >= 0 && ixy12 < src_size)
                    fxy2 += (x2 - x) * src[ixy12];
                if(ixy22 >= 0 && ixy22 < src_size)
                    fxy2 += (x - x1) * src[ixy22];
                dst[rz] += (y2 - y) * fxy1 + (y - y1) * fxy2;
            }
        }
    }
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
array_t<_Tp>
cxx_rotate(const _Tp* src, float theta, const intmax_t& nx, const intmax_t& ny, int = 0,
           double scale = 1.0)
{
    array_t<_Tp> dst(nx * ny, _Tp());
    cxx_rotate_ip(dst, src, theta, nx, ny, 0, scale);
    return dst;
}
#endif

//======================================================================================//
