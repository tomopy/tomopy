// Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.

// Copyright 2015. UChicago Argonne, LLC. This software was produced
// under U.S. Government contract DE-AC02-06CH11357 for Argonne National
// Laboratory (ANL), which is operated by UChicago Argonne, LLC for the
// U.S. Department of Energy. The U.S. Government has rights to use,
// reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR
// UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
// ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is
// modified to produce derivative works, such modified software should
// be clearly marked, so as not to confuse it with the version available
// from ANL.

// Additionally, redistribution and use in source and binary forms, with
// or without modification, are permitted provided that the following
// conditions are met:

//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.

//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in
//       the documentation and/or other materials provided with the
//       distribution.

//     * Neither the name of UChicago Argonne, LLC, Argonne National
//       Laboratory, ANL, the U.S. Government, nor the names of its
//       contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago
// Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "utils.hh"
#include "common.hh"

#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#if defined(TOMOPY_USE_OPENCV)
#    include <opencv2/highgui/highgui.hpp>
#    include <opencv2/imgcodecs.hpp>
#    include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
#endif

//======================================================================================//

template <typename _Tp> using Vector = std::vector<_Tp>;

//======================================================================================//

farray_t
cxx_expand(const float* src, const int factor, const int nx, const int ny)
{
    farray_t    dst  = farray_t(nx * factor * ny * factor, 0.0f);
    const float mult = factor * factor;
    const int   size = nx * ny;
    for(int j = 0; j < ny; ++j)
    {
        for(int i = 0; i < nx * ny; ++i)
        {
            int   idx00 = j * nx + i;
            float val   = src[idx00] / mult;
            dst[idx00]  = val;
            for(int off = 1; off <= factor; ++off)
            {
                int idx10 = j * nx + (i + off);
                int idx01 = (j + off) * nx + i;
                int idx11 = (j + off) * nx + (i + off);
                if(idx10 < size)
                    dst[idx10] = val;
                if(idx01 < size)
                    dst[idx01] = val;
                if(idx11 < size)
                    dst[idx11] = val;
            }
        }
    }
    return dst;
}

//======================================================================================//

farray_t
cxx_compress(const float* src, const int factor, const int nx, const int ny)
{
    farray_t  dst  = farray_t(nx * ny, 0.0f);
    const int size = abs(factor) * nx * abs(factor) * ny;
    for(int j = 0; j < ny; ++j)
    {
        for(int i = 0; i < nx * ny; ++i)
        {
            int idx00  = j * nx + i;
            dst[idx00] = src[idx00];
            for(int off = 1; off <= abs(factor); ++off)
            {
                int idx10 = j * nx + (i + off);
                int idx01 = (j + off) * nx + i;
                int idx11 = (j + off) * nx + (i + off);
                if(idx10 < size)
                    dst[idx00] += src[idx10];
                if(idx01 < size)
                    dst[idx00] += src[idx01];
                if(idx11 < size)
                    dst[idx00] += src[idx11];
            }
        }
    }
    return dst;
}

//======================================================================================//
#if defined(TOMOPY_USE_OPENCV)
Mat
cxx_affine_transform(const Mat& warp_src, float theta, const int nx, const int ny,
                     float scale)
{
    Mat   warp_dst = Mat::zeros(nx, ny, warp_src.type());
    float cx       = 0.5f * ny + ((ny % 2 == 0) ? 0.5f : 0.0f);
    float cy       = 0.5f * nx + ((nx % 2 == 0) ? 0.5f : 0.0f);
    Point center   = Point(cx, cy);
    Mat   rot      = getRotationMatrix2D(center, theta, scale);
    warpAffine(warp_src, warp_dst, rot, warp_src.size(), INTER_CUBIC);
    return warp_dst;
}
#endif
//======================================================================================//

void
cxx_affine_transform(farray_t& dst, const float* src, float theta, const int nx,
                     const int ny, const int factor)
{
#if defined(TOMOPY_USE_OPENCV)
    if(factor > 1)
    {
        dst          = std::move(cxx_expand(src, factor, nx, ny));
        int dx       = nx * factor;
        int dy       = ny * factor;
        Mat warp_src = Mat::zeros(dx, dy, CV_32F);
        memcpy(warp_src.ptr(), dst.data(), dst.size() * sizeof(float));
        float angle    = theta * (180.0f / pi);
        float scale    = 1.0;
        Mat   warp_rot = cxx_affine_transform(warp_src, angle, dx, dy, scale);
        memcpy(dst.data(), warp_rot.ptr(), dx * dy * sizeof(float));
    }
    else if(factor < -1)
    {
        int dx       = nx * abs(factor);
        int dy       = ny * abs(factor);
        Mat warp_src = Mat::zeros(dx, dy, CV_32F);
        memcpy(warp_src.ptr(), src, dx * dy * sizeof(float));
        float angle    = theta * (180.0f / pi);
        float scale    = 1.0;
        Mat   warp_rot = cxx_affine_transform(warp_src, angle, dx, dy, scale);
        dst.resize(dx * dy, 0.0f);
        memcpy(dst.data(), warp_rot.ptr(), dx * dy * sizeof(float));
        dst = std::move(cxx_compress(dst.data(), abs(factor), nx, ny));
    }
    else
    {
        Mat warp_src = Mat::zeros(nx, ny, CV_32F);
        memcpy(warp_src.ptr(), src, nx * ny * sizeof(float));
        float angle    = theta * (180.0f / pi);
        float scale    = 1.0;
        Mat   warp_rot = cxx_affine_transform(warp_src, angle, nx, ny, scale);
        memcpy(dst.data(), warp_rot.ptr(), nx * ny * sizeof(float));
    }
#else
    std::stringstream ss;
    ss << __FUNCTION__ << " not implemented without OpenCV!";
    throw std::runtime_error(ss.str());
#endif
}

//======================================================================================//

float
bilinear_interpolation(float x, float y, float x1, float x2, float y1, float y2,
                       float x1y1, float x2y1, float x1y2, float x2y2)
{
    if(x1 > x2)
        std::swap(x1, x2);
    if(y1 > y2)
        std::swap(y1, y2);

    if(x < x1 || x > x2)
        printf(
            "Warning! interpolating for x = %6.3f which is not in bound [%6.3f, %6.3f]\n",
            x, x1, x2);
    if(y < y1 || y > y2)
        printf(
            "Warning! interpolating for y = %6.3f which is not in bound [%6.3f, %6.3f]\n",
            y, y1, y2);

    printf("interpolating for x = %6.3f from [%6.3f, %6.3f]. values = [%6.3f, %6.3f]\n",
           x, x1, x2, x1y1, x2y1);
    printf("interpolating for x = %6.3f from [%6.3f, %6.3f]. values = [%6.3f, %6.3f]\n",
           x, x1, x2, x1y2, x2y2);

    float fx1y1 = (x2 - x) / (x2 - x1) * x1y1;
    float fx2y1 = (x - x1) / (x2 - x1) * x2y1;
    float fx1y2 = (x2 - x) / (x2 - x1) * x1y2;
    float fx2y2 = (x - x1) / (x2 - x1) * x2y2;

    printf("interpolating for y = %6.3f from [%6.3f, %6.3f]. values = [%6.3f, %6.3f]\n",
           y, y1, y2, (fx1y1 + fx2y1), (fx1y2 + fx2y2));
    printf("%8.3f, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f\n", fx1y1, fx2y1, fx1y2, fx2y2,
           (fx1y1 + fx2y1), (fx1y2 + fx2y2));

    return (y2 - y) / (y2 - y1) * (fx1y1 + fx2y1) +
           (y - y1) / (y2 - y1) * (fx1y2 + fx2y2);
}

//======================================================================================//

farray_t
cxx_rotate(const float* src, float theta, const int nx, const int ny)
{
    farray_t dst(nx * ny, 0.0);
    cxx_rotate_ip(dst, src, theta, nx, ny);
    return dst;
}

//======================================================================================//

void
cxx_rotate_ip(farray_t& dst, const float* src, float theta, const int nx, const int ny)
{
    memset(dst.data(), 0, nx * ny * sizeof(float));
#if defined(TOMOPY_USE_OPENCV)
    cxx_affine_transform(dst, src, theta, nx, ny, 1);
#else
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
#endif
    return;
}

//======================================================================================//
