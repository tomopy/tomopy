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

#include <complex.h>
#include <stdlib.h>

#ifdef WIN32
#    define DLL __declspec(dllexport)
#else
#    define DLL
#endif
#define ANSI

#if defined(TOMOPY_CXX_GRIDREC) && defined(WIN32)
#    define _Complex
#endif

DLL void
gridrec(const float* data, int dy, int dt, int dx, const float* center,
        const float* theta, float* recon, int ngridx, int ngridy, const char fname[16],
        const float* filter_par);

float*
malloc_vector_f(size_t n);

void
free_vector_f(float* v);

float _Complex*
malloc_vector_c(size_t n);

void
free_vector_c(float _Complex* v);

float _Complex**
malloc_matrix_c(size_t nr, size_t nc);

void
free_matrix_c(float _Complex** m);

float (*get_filter(const char* name))(float, int, int, int, const float*);

float
filter_none(float, int, int, int, const float*);

float
filter_shepp(float, int, int, int, const float*);

float
filter_hann(float, int, int, int, const float*);

float
filter_hamming(float, int, int, int, const float*);

float
filter_ramlak(float, int, int, int, const float*);

float
filter_parzen(float, int, int, int, const float*);

float
filter_butterworth(float, int, int, int, const float*);

float
filter_custom(float, int, int, int, const float*);

float
filter_custom2d(float, int, int, int, const float*);

unsigned char
filter_is_2d(const char* name);

void
set_filter_tables(int dt, int pd, float fac,
                  float (*const pf)(float, int, int, int, const float*),
                  const float* filter_par, float _Complex* A, unsigned char is2d);

void
set_trig_tables(int dt, const float* theta, float** SP, float** CP);

void
set_pswf_tables(float C, int nt, float lmbda, const float* coefs, int ltbl, int linv,
                float* wtbl, float* winv);

float
legendre(int n, const float* coefs, float x);

extern DLL void
cxx_gridrec(const float* data, int dy, int dt, int dx, const float* center,
            const float* theta, float* recon, int ngridx, int ngridy,
            const char fname[16], const float* filter_par);
