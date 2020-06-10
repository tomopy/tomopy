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

#pragma once

#include "utils.hh"

#include <complex.h>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <math.h>

//===========================================================================//

BEGIN_EXTERN_C
#include "gridrec.h"
END_EXTERN_C

//===========================================================================//

#ifndef M_PI
#    define M_PI 3.14159265359
#endif

#define __LIKELY(x) __builtin_expect(!!(x), 1)
#ifdef __INTEL_COMPILER
#    define __PRAGMA_SIMD _Pragma("simd assert")
#    define __PRAGMA_SIMD_VECREMAINDER _Pragma("simd assert, vecremainder")
#    define __PRAGMA_SIMD_VECREMAINDER_VECLEN8                                           \
        _Pragma("simd assert, vecremainder, vectorlength(8)")
#    define __PRAGMA_OMP_SIMD_COLLAPSE _Pragma("omp simd collapse(2)")
#    define __PRAGMA_IVDEP _Pragma("ivdep")
#    define __ASSSUME_64BYTES_ALIGNED(x) __assume_aligned((x), 64)
#else
#    define __PRAGMA_SIMD
#    define __PRAGMA_SIMD_VECREMAINDER
#    define __PRAGMA_SIMD_VECREMAINDER_VECLEN8
#    define __PRAGMA_OMP_SIMD_COLLAPSE
#    define __PRAGMA_IVDEP
#    define __ASSSUME_64BYTES_ALIGNED(x)
#endif

//===========================================================================//

typedef std::function<float(float, int, int, int, const float*)> filter_func;

//===========================================================================//

float*
cxx_malloc_vector_f(size_t n);

void
cxx_free_vector_f(float*& v);

std::complex<float>*
cxx_malloc_vector_c(size_t n);

void
cxx_free_vector_c(std::complex<float>*& v);

std::complex<float>**
cxx_malloc_matrix_c(size_t nr, size_t nc);

void
cxx_free_matrix_c(std::complex<float>**& m);

void
cxx_set_filter_tables(int dt, int pd, float fac, filter_func, const float* filter_par,
                      std::complex<float>* A, unsigned char is2d);

//===========================================================================//
