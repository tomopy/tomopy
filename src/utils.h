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

#ifndef _utils_h
#define _utils_h

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <sys/time.h>
#include <assert.h>
#include "string.h"


#define _USE_MATH_DEFINES
#ifndef M_PI
#define M_PI 3.14159265358979323846264338327
#endif

#ifdef WIN32
#define DLL __declspec(dllexport)
#else
#define DLL 
#endif

// Data simulation

void DLL
project(
    const float *obj,
    int oy,
    int ox, 
    int oz,
    float *data,
    int dy, 
    int dt,
    int dx,
    const float *center,
    const float *theta);

// Reconstruction algorithms

void DLL
art(
    const float *data,
    int dy, 
    int dt,
    int dx,
    const float *center,
    const float *theta,
    float *recon,
    int ngridx,
    int ngridy,
    int num_iter);

void DLL
bart(
    const float *data,
    int dy, 
    int dt,
    int dx,
    const float *center,
    const float *theta,
    float *recon,
    int ngridx,
    int ngridy,
    int num_iter,
    int num_block,
    const float *ind_block); //TODO: I think this should be int *

void DLL
fbp(
    const float *data,
    int dy,
    int dt,
    int dx,
    const float *center,
    const float *theta,
    float *recon,
    int ngridx,
    int ngridy,
<<<<<<< HEAD
    char name[16],
    float *filter_par,
    int istart, 
    int iend);
=======
    char name[16]);
>>>>>>> 397df9d1cfeb5f47215fd40a993486327b871411

void DLL
mlem(
    const float *data,
    int dy, 
    int dt,
    int dx,
    const float *center,
    const float *theta,
    float *recon,
    int ngridx,
    int ngridy,
    int num_iter);

void DLL
osem(
    const float *data,
    int dy, 
    int dt,
    int dx,
    const float *center,
    const float *theta,
    float *recon,
    int ngridx,
    int ngridy,
    int num_iter,
    int num_block,
    const float *ind_block);

void DLL
ospml_hybrid(
    const float *data,
    int dy, 
    int dt,
    int dx,
    const float *center,
    const float *theta,
    float *recon,
    int ngridx,
    int ngridy,
    int num_iter,
    const float *reg_pars,
    int num_block,
    const float *ind_block);

void DLL
ospml_quad(
    const float *data,
    int dy, 
    int dt,
    int dx,
    const float *center,
    const float *theta,
    float *recon,
    int ngridx,
    int ngridy,
    int num_iter,
    const float *reg_pars,
    int num_block,
    const float *ind_block);

void DLL
pml_hybrid(
    const float *data,
    int dy, 
    int dt,
    int dx,
    const float *center,
    const float *theta,
    float *recon,
    int ngridx,
    int ngridy,
    int num_iter,
    const float *reg_pars);

void DLL
pml_quad(
    const float *data,
    int dy, 
    int dt,
    int dx,
    const float *center,
    const float *theta,
    float *recon,
    int ngridx,
    int ngridy,
    int num_iter,
    const float *reg_pars);

void DLL
sirt(
    const float *data,
    int dy, 
    int dt,
    int dx,
    const float *center,
    const float *theta,
    float *recon,
    int ngridx,
    int ngridy,
    int num_iter);

// Utility functions for data simultation

void DLL
preprocessing(
    int ngridx, int ngridy, 
    int dz, 
    float center, float *mov, 
    float *gridx, float *gridy);

int DLL
calc_quadrant(
    float theta_p); 

void DLL
calc_coords(
    int ngridx, int ngridy,
    float xi, float yi,
    float sin_p, float cos_p,
    const float *gridx, const float *gridy,
    float *coordx, float *coordy);

void DLL
trim_coords(
    int ngridx, int ngridy,
    const float *coordx, const float *coordy,
    const float *gridx, const float *gridy,
    int *asize, float *ax, float *ay, 
    int *bsize, float *bx, float *by);

void DLL
sort_intersections(
    int ind_condition, 
    int asize, const float *ax, const float *ay,
    int bsize, const float *bx, const float *by,
    int *csize, 
    float *coorx, float *coory);

void DLL
calc_dist(
    int ngridx, int ngridy, 
    int csize, 
    const float *coorx, const float *coory,
    int *indi, 
    float *dist);

void DLL
calc_simdata(
    int s, int p, int d,
    int ngridx, int ngridy, 
    int dt, int dx,
    int csize, 
    const int *indi,
    const float *dist,
    const float *model,
    float *simdata);

#endif
