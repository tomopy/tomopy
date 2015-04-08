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


#define ANSI
#define max(A,B) ((A)>(B)?(A):(B))
#define min(A,B) ((A)<(B)?(A):(B))
#define free_matrix(A) (free(*(A)),free(A))
#define abs(A) ((A)>0 ?(A):-(A))
#define PI 3.14159265359
#define TOLERANCE 0.1
#define LTBL_DEF 512
#define LEN_FTBL 7
#define Cnvlvnt(X) (wtbl[(int)(X+0.5)])    
#define Cmult(A,B,C) {(A).r=(B).r*(C).r-(B).i*(C).i;\
             (A).i=(B).r*(C).i+(B).i*(C).r;}


// Data structures

typedef struct 
{
    float r; // Real part
    float i; // Imaginary part
} complex;

typedef struct 
{
    int dx;
    int dy;
    int dz;
    float center;
    float *proj_angle;
} data_pars;


// Prolate spheroidal wave function (PSWF) parameters 
typedef struct{ 
    float C; // Parameter for particular 0th order pswf being used 
    int nt; // Degree of Legendre polynomial expansion 
    float lmbda; // Eigenvalue 
    float coefs[15]; // Coefficients for Legendre polynomial expansion 
} pswf_struct;


// Parameters for gridrec algorithm 
typedef struct {
    pswf_struct *pswf; // Pointer to data for PSWF being used  
    float sampl; // "Oversampling" ratio 
    char fname[16]; // Name of filter function       
    float (*filter)(float); // Pointer to filter function 
} grid_pars;


typedef struct 
{
    int num_iter;
    float *reg_pars;
    int rx;
    int ry;
    int rz;
    float *ind_block;
    int num_block;
} recon_pars;

// Functions for regridding algorithm 

void gridrec_init(
    data_pars *dpar,
    grid_pars *gpar);

void gridrec_main(
    float *data, 
    data_pars *dpar, 
    float *recon, 
    recon_pars *rpar);

void gridrec(
    float *data, 
    data_pars *dpar, 
    float *recon, 
    grid_pars *gpar);

void gridrec_free(void);


float*** convert(float *arr, int dim0, int dim1, int dim2);
float (*get_filter(char *name))(float);
void get_pswf(float C, pswf_struct **P);


// FFT routines from Numerical Recipes

void 
four1(
    float data[], 
    unsigned long nn, 
    int isign);

void 
fourn(
    float data[], 
    unsigned long nn[], 
    int ndim, 
    int isign);

#endif
