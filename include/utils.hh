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

DLL void
cxx_affine_transform(farray_t& dst, const float* src, float theta_rad, float theta_deg,
                     const int nx, const int ny, const float scale = 1.0f);

//--------------------------------------------------------------------------------------//

DLL farray_t
    cxx_rotate(const float* src, float theta, const int nx, const int ny);

//--------------------------------------------------------------------------------------//

DLL void
cxx_rotate_ip(farray_t& dst, const float* src, float theta, const int nx, const int ny);

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
    cpu_memcpy<_Tp>(gpu_data, cpu_data.data(), n);
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

//--------------------------------------------------------------------------------------//
