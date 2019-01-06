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

#ifndef _utils_openacc_h
#define _utils_openacc_h

#include "string.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

#if !defined(DLL)
#    ifdef WIN32
#        define DLL __declspec(dllexport)
#    else
#        define DLL
#    endif
#endif

//============================================================================//

BEGIN_EXTERN_C

//============================================================================//

DLL void
openacc_preprocessing(int ngridx, int ngridy, int dz, float center, float* mov,
                      float* gridx, float* gridy);

int DLL
    openacc_calc_quadrant(float theta_p);

DLL void
openacc_calc_coords(int ngridx, int ngridy, float xi, float yi, float sin_p, float cos_p,
                    const float* gridx, const float* gridy, float* coordx, float* coordy);

DLL void
openacc_trim_coords(int ngridx, int ngridy, const float* coordx, const float* coordy,
                    const float* gridx, const float* gridy, int* asize, float* ax,
                    float* ay, int* bsize, float* bx, float* by);

DLL void
openacc_sort_intersections(int ind_condition, int asize, const float* ax, const float* ay,
                           int bsize, const float* bx, const float* by, int* csize,
                           float* coorx, float* coory);

DLL void
openacc_calc_sum_sqr(const int csize, const float* dist, float* sum_sqr);

DLL void
openacc_calc_dist(int ngridx, int ngridy, int csize, const float* coorx,
                  const float* coory, int* indi, float* dist);

DLL void
openacc_calc_simdata(int s, int p, int d, int ngridx, int ngridy, int dt, int dx,
                     int csize, const int* indi, const float* dist, const float* model,
                     float* simdata);

//============================================================================//

END_EXTERN_C

//============================================================================//

#endif
