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

#if defined(TOMOPY_USE_IPP)
#    include <ipp.h>
#    include <ippdefs.h>
#    include <ippi.h>
#endif

#if defined(TOMOPY_USE_OPENCV)
#    include <opencv2/highgui/highgui.hpp>
#    include <opencv2/imgcodecs.hpp>
#    include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
#endif

//======================================================================================//

template <typename _Tp>
using Vector = std::vector<_Tp>;

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
cxx_affine_transform(farray_t& dst, const float* src, float theta_rad, float theta_deg,
                     const int nx, const int ny, const float scale)
{
#if defined(TOMOPY_USE_OPENCV)

    Mat warp_src = Mat::zeros(nx, ny, CV_32F);
    memcpy(warp_src.ptr(), src, nx * ny * sizeof(float));
    Mat warp_rot = cxx_affine_transform(warp_src, theta_deg, nx, ny, scale);
    memcpy(dst.data(), warp_rot.ptr(), nx * ny * sizeof(float));

#elif defined(TOMOPY_USE_IPP)

    auto getRotationMatrix2D = [&](double m[2][3]) {
        double alpha    = scale * cos(theta_rad);
        double beta     = scale * sin(theta_rad);
        double center_x = (0.5 * nx) - 0.5;
        double center_y = (0.5 * ny) - 0.5;

        m[0][0] = alpha;
        m[0][1] = beta;
        m[0][2] = (1.0 - alpha) * center_x - beta * center_y;

        m[1][0] = -beta;
        m[1][1] = alpha;
        m[1][2] = beta * center_x + (1.0 - alpha) * center_y;
    };

    IppiSize siz;
    siz.width  = nx;
    siz.height = ny;

    /*IppiRect roi;
    roi.x      = 0;
    roi.y      = 0;
    roi.width  = nx;
    roi.height = ny;*/

    double        rot[2][3];
    int           bufSize   = 0;
    int           specSize  = 0;
    int           initSize  = 0;
    int           step      = nx * sizeof(float);
    IppiPoint     dstOffset = { 0, 0 };
    IppiWarpSpec* pSpec     = NULL;
    getRotationMatrix2D(rot);

    ippiWarpAffineGetSize(siz, siz, ipp32f, rot, ippCubic, ippWarpForward,
                          ippBorderDefault, &specSize, &initSize);
    ippiWarpGetBufferSize(pSpec, siz, &bufSize);
    pSpec          = (IppiWarpSpec*) ippsMalloc_32f(specSize);
    Ipp8u* pBuffer = ippsMalloc_8u(bufSize);

    ippiWarpAffineCubic_32f_C1R(src, step, dst.data(), step, dstOffset, siz, pSpec,
                                pBuffer);

    ippsFree(pSpec);
    ippsFree(pBuffer);

#else

    std::stringstream ss;
    ss << __FUNCTION__ << " not implemented without OpenCV or Intel IPP!";
    throw std::runtime_error(ss.str());

#endif
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
#if defined(TOMOPY_USE_OPENCV) || defined(TOMOPY_USE_IPP)
    cxx_affine_transform(dst, src, theta, theta * degrees, nx, ny, 1);
#else

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
#endif
    return;
}

//======================================================================================//
