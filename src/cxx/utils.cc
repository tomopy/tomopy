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

//============================================================================//

template <typename _Tp> using Vector = std::vector<_Tp>;

//============================================================================//

namespace
{
constexpr float pi       = (float) M_PI;
constexpr float halfpi   = 0.5f * pi;
constexpr float twopi    = 2.0f * pi;
constexpr float epsilonf = 2.0f * std::numeric_limits<float>::epsilon();
}

//============================================================================//

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

//============================================================================//

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

//============================================================================//
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
//============================================================================//

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

//============================================================================//

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

//============================================================================//

farray_t
cxx_apply_rotation(const float* src, float theta, const int nx, const int ny)
{
    farray_t dst(nx * ny, 0.0);
    cxx_apply_rotation_ip(dst, src, theta, nx, ny);
    return dst;
}

//============================================================================//

farray_t
cxx_remove_rotation(const float* src, float theta, const int nx, const int ny)
{
    farray_t dst(nx * ny, 0.0);
    cxx_remove_rotation_ip(dst, src, theta, nx, ny);
    return dst;
}

//============================================================================//

void
cxx_apply_rotation_ip(farray_t& dst, const float* src, float theta, const int nx,
                      const int ny)
{
    memset(dst.data(), 0, nx * ny * sizeof(float));
#if defined(TOMOPY_USE_OPENCV)
    cxx_affine_transform(dst, src, theta, nx, ny, 1);
#else
    cxx_remove_rotation_ip(dst, src, theta, nx, ny);
#endif
    return;
}

//============================================================================//

struct Pixel
{
    float x;
    float y;
    float p;

    Pixel()
    : x(0.0f)
    , y(0.0f)
    , p(0.0f)
    {
    }
    Pixel(const farray_t& rhs)
    : x(rhs[0])
    , y(rhs[1])
    , p(rhs[2])
    {
    }

    ~Pixel()            = default;
    Pixel(const Pixel&) = default;
    Pixel(Pixel&&)      = default;
    Pixel& operator=(const Pixel&) = default;
    Pixel& operator=(Pixel&&) = default;

    Pixel& operator=(const farray_t& rhs)
    {
        x = rhs[0];
        y = rhs[1];
        p = rhs[2];
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const Pixel& pixel)
    {
        std::stringstream ss;
        int               p = 2;
        int               w = 4;
        ss.precision(p);
        ss << "[" << std::setw(w) << pixel.x << ", " << std::setw(w) << pixel.y
           << "] = " << std::setw(w) << pixel.p;
        os << ss.str();
        return os;
    }

    Pixel& operator+=(const float& rhs)
    {
        x += rhs;
        y += rhs;
        return *this;
    }
    Pixel operator+(const float& rhs) { return Pixel(*this) += rhs; }

    Pixel& operator-=(const float& rhs)
    {
        x -= rhs;
        y -= rhs;
        return *this;
    }
    Pixel operator-(const float& rhs) { return Pixel(*this) -= rhs; }
};

struct SortedPixels : public std::vector<Pixel>
{
    SortedPixels(const Vector<farray_t>& rhs, const int& nx, const int ny, const float&,
                 const float&)
    {
        for(const auto& itr : rhs)
        {
            float _x = itr[0];
            float _y = itr[1];
            if(_x > 0.0f && _y > 0.0f && _x < nx && _y < ny)
                push_back(Pixel(itr));
        }
        std::sort(begin(), end(),
                  [](const Pixel& lhs, const Pixel& rhs) { return (lhs.y < rhs.y); });
        std::sort(begin(), end(), [](const Pixel& lhs, const Pixel& rhs) {
            return (lhs.x == rhs.x) ? (lhs.y < rhs.y) : (lhs.x < rhs.x);
        });
        std::cout << "Sorted pixels: " << std::endl;
        for(const auto& itr : *this)
            std::cout << "\t" << itr << std::endl;
    }
};

struct PixelNeighbors
{
    Pixel                           center;
    bool                            exact = false;
    typedef std::pair<bool, Pixel>  neighbor_t;
    typedef Vector<neighbor_t>      neighbor_list_t;
    typedef Vector<neighbor_list_t> neighbor_matrix_t;

    neighbor_matrix_t neighbors =
        neighbor_matrix_t(2, neighbor_list_t(2, neighbor_t(false, Pixel())));
    Pixel& x1y1 = neighbors[0][0].second;
    Pixel& x2y1 = neighbors[0][1].second;
    Pixel& x1y2 = neighbors[1][0].second;
    Pixel& x2y2 = neighbors[1][1].second;

    PixelNeighbors()                      = default;
    ~PixelNeighbors()                     = default;
    PixelNeighbors(const PixelNeighbors&) = default;
    PixelNeighbors(PixelNeighbors&&)      = default;
    PixelNeighbors& operator=(const PixelNeighbors&) = default;
    PixelNeighbors& operator=(PixelNeighbors&&) = default;

    friend std::ostream& operator<<(std::ostream& os, const PixelNeighbors& rhs)
    {
        std::stringstream ss;
        int               p = 2;
        int               w = 8;
        ss.precision(p);
        ss << std::fixed;
        ss << std::setw(w) << "center " << rhs.center << std::endl;
        if(rhs.neighbors[0][0].first)
            ss << std::setw(w) << "x1y1 " << rhs.neighbors[0][0].second << std::endl;
        if(rhs.neighbors[0][1].first)
            ss << std::setw(w) << "x2y1 " << rhs.neighbors[0][1].second << std::endl;
        if(rhs.neighbors[1][0].first)
            ss << std::setw(w) << "x1y2 " << rhs.neighbors[1][0].second << std::endl;
        if(rhs.neighbors[1][1].first)
            ss << std::setw(w) << "x2y2 " << rhs.neighbors[1][1].second << std::endl;
        os << ss.str();
        return os;
    }

    float interpolate()
    {
        float fxy = center.p;
        if(exact)
            return fxy;
        int count = 0;
        for(int i = 0; i < 2; ++i)
            for(int j = 0; j < 2; ++j)
                if(neighbors[j][i].first)
                    count++;

        if(count > 2)
        {
            fxy = bilinear_interpolation(center.x, center.y, x1y1.x, x2y1.x, x2y1.y,
                                         x2y2.y, x1y1.p, x2y1.p, x2y1.p, x2y2.p);
        }
        else if(count == 2)
        {
            auto interpolate = [=](float x, float x1, float x2, float y1, float y2) {
                return y1 + (y2 - y1) * (x - x1) / (x2 - x1);
            };

            if(neighbors[0][0].first && neighbors[0][1].first)
                fxy = interpolate(center.x, x1y1.x, x2y1.x, x1y1.p, x2y1.p);
            else if(neighbors[1][0].first && neighbors[1][1].first)
                fxy = interpolate(center.x, x1y2.x, x2y2.x, x1y2.p, x2y2.p);
            else if(neighbors[0][0].first && neighbors[1][0].first)
                fxy = interpolate(center.x, x1y1.x, x1y2.x, x1y1.p, x1y2.p);
            else if(neighbors[0][1].first && neighbors[1][1].first)
                fxy = interpolate(center.x, x2y1.x, x2y2.x, x2y1.p, x2y2.p);
        }
        else if(count == 1)
        {
            for(int i = 0; i < 2; ++i)
                for(int j = 0; j < 2; ++j)
                    if(neighbors[j][i].first)
                        return neighbors[j][i].second.p;
        }
        return fxy;
    }

    void find(const SortedPixels& sorted_coords)
    {
        for(const auto& itr : sorted_coords)
        {
            float dx = center.x - itr.x;
            float dy = center.y - itr.y;

            std::cout.precision(2);
            std::cout << "\t center " << center << ", itr " << itr
                      << ", dx = " << std::setw(4) << dx << ", dy = " << std::setw(4)
                      << dy << std::endl;

            if(fabs(dx) < epsilonf && fabs(dy) < epsilonf)
            {
                center.p = itr.p;
                exact    = true;
                printf("\t\tSkipping...\n");
                break;
            }

            auto update = [&](int _x, int _y) {
                float dist = sqrtf(powf(center.x - neighbors[_y][_x].second.x, 2.0f) +
                                   powf(center.y - neighbors[_y][_x].second.x, 2.0f));
                if(!neighbors[_y][_x].first || sqrtf(dx * dx + dy * dy) < dist)
                {
                    std::cout << "\t\tSetting x" << _x + 1 << "y" << _y + 1 << itr
                              << "..." << std::endl;
                    neighbors[_y][_x].first  = true;
                    neighbors[_y][_x].second = itr;
                }
            };

            if(dx > 0.0f && dy > 0.0f && itr.x < center.x && itr.y < center.y)
            {
                update(0, 0);
            }
            if(dx > 0.0f && dy < 0.0f && itr.x < center.x && itr.y > center.y)
            {
                update(0, 1);
            }
            if(dx < 0.0f && dy > 0.0f && itr.x > center.x && itr.y < center.y)
            {
                update(1, 0);
            }
            if(dx < 0.0f && dy < 0.0f && itr.x > center.x && itr.y > center.y)
            {
                update(1, 1);
            }
        }
    }
};

//============================================================================//

void
cxx_remove_rotation_ip(farray_t& dst, const float* src, float theta, const int nx,
                       const int ny)
{
    memset(dst.data(), 0, nx * ny * sizeof(float));
#if defined(TOMOPY_USE_OPENCV)
    cxx_affine_transform(dst, src, theta, nx, ny, 1);
#else

    static thread_local int verbose = GetEnv<int>("TOMOPY_VERBOSE", 0);
    float                   dsum    = 0.0f;
    float                   ssum    = 0.0f;
    theta                           = fmodf(theta, twopi);
    float ptheta                    = (theta < 0.0f) ? twopi + theta : theta;
    float xoff                      = truncf(nx / 2.0f);
    float yoff                      = truncf(ny / 2.0f);
    float xop                       = (nx % 2 == 0) ? 0.5f : 0.0f;
    float yop                       = (ny % 2 == 0) ? 0.5f : 0.0f;
    int   xcos                      = (ptheta > halfpi && ptheta < 1.5f * halfpi) ? 1 : 1;
    int   ysin                      = (ptheta > pi) ? 1 : 1;
    int   size                      = nx * ny;
    float epsilon                   = 2.0f * std::numeric_limits<float>::epsilon();

    std::vector<farray_t> coord(nx * ny, farray_t(3, 0.0f));
    std::vector<farray_t> rot   = { { cosf(theta), sinf(theta), 0.0f },
                                  { -sinf(theta), cosf(theta), 0.0f },
                                  { 0.0f, 0.0f, 1.0f } };
    farray_t              czero = { 0.0f, 0.0f, 0.0f };

    auto bilinear = [=](float x, float x1, float x2, float y, float y1, float y2,
                        float& fxy1, float& fxy2) {
        // lambda that computes the source value for given indices
        auto compute = [=](int _x, int _y) {
            auto _f = (_x > 0 && _y > 0 && _x < nx && _y < ny && _y * nx + _x < size)
                          ? coord[_y * nx + _x]
                          : czero;
            return _f;
        };
        auto _rdn = [=](const float& _f) {
            return static_cast<int>(floorf(_f) + epsilon);
        };
        auto _rup = [=](const float& _f) {
            return static_cast<int>(ceilf(_f) + epsilon);
        };

        fxy1 += (x2 - x) * compute(_rdn(x1), _rdn(y1))[0];
        fxy1 += (x - x1) * compute(_rup(x2), _rdn(y1))[0];
        fxy2 += (x2 - x) * compute(_rdn(x1), _rup(y2))[1];
        fxy2 += (x - x1) * compute(_rup(x2), _rup(y2))[1];
        return (y2 - y) * fxy1 + (y - y1) * fxy2;
    };
    auto fetch = [=](int _x, int _y) {
        farray_t _f = (_x > 0 && _y > 0 && _x < nx && _y < ny && _y * nx + _x < size)
                          ? coord[_y * nx + _x]
                          : farray_t({ _x + 0.5f, _y + 0.5f, 0.0f });
        return _f;
    };
    auto update = [&](int _x, int _y, const float& pix) {
        if(_x >= 0 && _y >= 0 && _x < nx && _y < ny && _y * nx + _x < size)
        {
            dst[_y * nx + _x] += pix;
            dsum += pix;
        }
    };
    /*
    auto interpolate = [=](float x, float x1, float x2, float y1, float y2) {
        return y1 + (y2 - y1) * (x - x1) / (x2 - x1);
    };
    */

    for(int j = 0; j < ny; ++j)
    {
        for(int i = 0; i < nx; ++i)
        {
            // offset indices
            float pixel = src[j * nx + i];
            float ox    = float(i) - xoff + xop;
            float oy    = float(j) - yoff + yop;
            ssum += pixel;
            farray_t  row    = { ox, oy, pixel };
            farray_t& _coord = coord[j * nx + i];
            for(int jj = 0; jj < 3; ++jj)
            {
#    pragma omp simd
                for(int ii = 0; ii < 3; ++ii)
                {
                    _coord[jj] += row[ii] * rot[jj][ii];
                }
            }
            _coord[0] += xoff;
            _coord[1] += yoff;
        }
    }

    SortedPixels sorted_pixels(coord, nx, ny, xop, yop);

    std::ofstream ofs("pixels.txt");
    for(const auto& itr : sorted_pixels)
    {
        ofs << itr.x << ", " << itr.y << ", " << itr.p << std::endl;
    }

    for(int j = 0; j < ny; ++j)
    {
        for(int i = 0; i < nx; ++i)
        {
            PixelNeighbors pn;
            pn.center = Pixel({ float(i) + 0.5f, float(j) + 0.5f, 0.0f });
            pn.find(sorted_pixels);

            std::cout << pn << std::endl;
            float fxy = pn.interpolate();
            if(std::isfinite(fxy))
                update(i, j, fxy);
        }
    }
    /*
    float scale = (fabs(dsum) > 0.0f) ? ssum / dsum : 1.0f;
    for(int j = 0; j < ny; ++j)
        for(int i = 0; i < nx; ++i)
            dst[j * nx + i] *= scale;
    */
#endif
}

//============================================================================//
