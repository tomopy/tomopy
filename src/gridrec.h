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

#ifndef _gridrec_h
#define _gridrec_h

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <time.h>
#include <sys/stat.h>


#ifdef WIN32
#define DLL __declspec(dllexport)
#else
#define DLL 
#endif
#define ANSI
#define max(A,B) ((A)>(B)?(A):(B))
#define min(A,B) ((A)<(B)?(A):(B))
#define free_matrix(A) (free(*(A)),free(A))
#define abs(A) ((A)>0 ?(A):-(A))
#define PI 3.14159265359
#define Cnvlvnt(X) (wtbl[(int)(X+0.5)])    
#define Cmult(A,B,C) {(A).r=(B).r*(C).r-(B).i*(C).i;\
             (A).i=(B).r*(C).i+(B).i*(C).r;}


typedef struct {
    float r;
    float i;
} complex;

void 
gridrec(
    float *data,
    int dx, int dy, int dz,
    float center,
    float *theta,
    float *recon,
    int ngridx, int ngridy,
    char name[16]);

float*** 
convert(float *arr, int dim0, int dim1, int dim2);

float* 
malloc_vector_f(long n);

complex* 
malloc_vector_c(long n);

complex**
malloc_matrix_c(long nr, long nc);

float 
(*get_filter(char *name))(float);

float 
filter_none(float);

float 
filter_shepp(float);

float 
filter_hann(float);

float 
filter_hamming(float);

float 
filter_ramlak(float);

void 
set_filter_tables(
    int dx, int pd, 
    float fac, float(*pf)(float), 
    complex *A);

void 
set_trig_tables(
    int dx, float *theta, 
    float **SP, float **CP);

void 
set_pswf_tables(
    float C, int nt, float lmbda, float *coefs, 
    int ltbl, int linv, float* wtbl, float* winv);

float 
legendre(int n, float *coefs, float x);

#endif
