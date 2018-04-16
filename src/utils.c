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

#include "utils.h"

//for windows build
#ifdef WIN32
	#ifdef PY3K
	void PyInit_libtomopy(void){}
	#else
	void initlibtomopy(void){}
	#endif
#endif


void 
preprocessing(
    int ry, int rz,
    int num_pixels, float center, 
    float *mov, float *gridx, float *gridy) 
{
    int i;

    for(i=0; i<=ry; i++)
    {
        gridx[i] = -ry/2.+i;
    }

    for(i=0; i<=rz; i++)
    {
        gridy[i] = -rz/2.+i;
    }

    *mov = ((float)num_pixels-1)/2.0-center;
    if(*mov-floor(*mov) < 0.01) {
        *mov += 0.01;
    }
    *mov += 0.5;
}


int 
calc_quadrant(
    float theta_p) 
{
    int quadrant;
    if ((theta_p >= 0 && theta_p < M_PI/2) ||
        (theta_p >= M_PI && theta_p < 3*M_PI/2) ||
        (theta_p >= -M_PI && theta_p < -M_PI/2) ||
        (theta_p >= -2*M_PI && theta_p < -3*M_PI/2)) 
    {
        quadrant = 1;
    } 
    else 
    {
        quadrant = 0;
    }
    return quadrant;
}


void 
calc_coords(
    int ry, int rz,
    float xi, float yi,
    float sin_p, float cos_p,
    const float *gridx, const float *gridy,
    float *coordx, float *coordy)
{
    float srcx, srcy, detx, dety;
    float slope, islope;
    int n;

    srcx = xi*cos_p-yi*sin_p;
    srcy = xi*sin_p+yi*cos_p;
    detx = -xi*cos_p-yi*sin_p;
    dety = -xi*sin_p+yi*cos_p;

    slope = (srcy-dety)/(srcx-detx);
    islope = 1/slope;
    for (n=0; n<=ry; n++) 
    {
        coordy[n] = slope*(gridx[n]-srcx)+srcy;
    }
    for (n=0; n<=rz; n++) 
    {
        coordx[n] = islope*(gridy[n]-srcy)+srcx;
    }
}


void 
trim_coords(
    int ry, int rz,
    const float *coordx, const float *coordy,
    const float *gridx, const float* gridy,
    int *asize, float *ax, float *ay, 
    int *bsize, float *bx, float *by)
{
    int n;

    *asize = 0;
    *bsize = 0;
    for (n=0; n<=rz; n++) 
    {
        if (coordx[n] >= gridx[0]+1e-2) 
        {
            if (coordx[n] <= gridx[ry]-1e-2) 
            {        
                ax[*asize] = coordx[n];
                ay[*asize] = gridy[n];
                (*asize)++;
            }
        }
    }
    for (n=0; n<=ry; n++) 
    {
        if (coordy[n] >= gridy[0]+1e-2) 
        {
            if (coordy[n] <= gridy[rz]-1e-2) 
            {
                bx[*bsize] = gridx[n];
                by[*bsize] = coordy[n];
                (*bsize)++;
            }
        }
    }
}


void 
sort_intersections(
    int ind_condition, 
    int asize, const float *ax, const float *ay,
    int bsize, const float *bx, const float *by,
    int *csize, float *coorx, float *coory)
{
    int i=0, j=0, k=0;
    int a_ind;
    while (i<asize && j<bsize)
    {
        a_ind = (ind_condition) ? i : (asize-1-i);
        if (ax[a_ind] < bx[j]) 
        {
            coorx[k] = ax[a_ind];
            coory[k] = ay[a_ind];
            i++;
            k++;
        } 
        else 
        {
            coorx[k] = bx[j];
            coory[k] = by[j];
            j++;
            k++;
        }
    }
    while (i < asize) 
    {
        a_ind = (ind_condition) ? i : (asize-1-i);
        coorx[k] = ax[a_ind];
        coory[k] = ay[a_ind];
        i++;
        k++;
    }
    while (j < bsize) 
    {
        coorx[k] = bx[j];
        coory[k] = by[j];
        j++;
        k++;
    }
    *csize = asize+bsize;
}


void 
calc_dist(
    int ry, int rz, 
    int csize, const float *coorx, const float *coory,
    int *indi, float *dist)
{
    int n, i1, i2;
    float x1, x2;
    float diffx, diffy, midx, midy;
    int indx, indy;

    for (n=0; n<csize-1; n++) 
    {
        diffx = coorx[n+1]-coorx[n];
        diffy = coory[n+1]-coory[n];
        dist[n] = sqrt(diffx*diffx+diffy*diffy);
        midx = (coorx[n+1]+coorx[n])/2;
        midy = (coory[n+1]+coory[n])/2;
        x1 = midx+ry/2.;
        x2 = midy+rz/2.;
        i1 = (int)(midx+ry/2.);
        i2 = (int)(midy+rz/2.);
        indx = i1-(i1>x1);
        indy = i2-(i2>x2);
        indi[n] = indy+(indx*rz);
    }
}


void 
calc_dist2(
    int ry, int rz, 
    int csize, const float *coorx, const float *coory,
    int *indx, int *indy, float *dist)
{
    int n, i1, i2;
    float x1, x2;
    float diffx, diffy, midx, midy;

    for (n=0; n<csize-1; n++) 
    {
        diffx = coorx[n+1]-coorx[n];
        diffy = coory[n+1]-coory[n];
        dist[n] = sqrt(diffx*diffx+diffy*diffy);
        midx = (coorx[n+1]+coorx[n])/2;
        midy = (coory[n+1]+coory[n])/2;
        x1 = midx+ry/2.;
        x2 = midy+rz/2.;
        i1 = (int)(midx+ry/2.);
        i2 = (int)(midy+rz/2.);
        indx[n] = i1-(i1>x1);
        indy[n] = i2-(i2>x2);
    }
}


void 
calc_simdata(
    int s, int p, int d,
    int ry, int rz, 
    int dt, int dx,
    int csize, const int *indi, const float *dist,
    const float *model, float *simdata)
{
    int n;

    int index_model = s*ry*rz;
    int index_data = d+p*dx+s*dt*dx;
    for (n=0; n<csize-1; n++) 
    {
        simdata[index_data] += model[indi[n]+index_model]*dist[n];
    }
}


void 
calc_simdata2(
    int s, int p, int d,
    int ry, int rz, 
    int dt, int dx,
    int csize, const int *indx, const int *indy, const float *dist,
    const float *model, int axis, float *simdata)
{
    int n;

    if (axis==0)
    {
        for (n=0; n<csize-1; n++) 
        {
            simdata[d + p*dx + s*dt*dx] += model[indy[n] + indx[n]*rz + s*ry*rz]*dist[n];
        }
    }
    else if (axis==1)
    {
        for (n=0; n<csize-1; n++) 
        {
            simdata[d + p*dx + s*dt*dx] += model[s + indx[n]*rz + indy[n]*ry*rz]*dist[n];
        }
    }
    else if (axis==2)
    {
        for (n=0; n<csize-1; n++) 
        {
            simdata[d + p*dx + s*dt*dx] += model[indx[n] + s*rz + indy[n]*ry*rz]*dist[n];
        }
    }
}


void 
calc_simdata3(
    int s, int p, int d,
    int ry, int rz, 
    int dt, int dx,
    int csize, const int *indx, const int *indy, const float *dist,
    float vx, float vy, 
    const float *modelx, const float *modely, const float *modelz, int axis, float *simdata)
{
    int n;

    if (axis==0)
    {
        for (n=0; n<csize-1; n++) 
        {
            simdata[d + p*dx + s*dt*dx] += (modelx[indy[n] + indx[n]*rz + s*ry*rz] * vx + modely[indy[n] + indx[n]*rz + s*ry*rz] * vy) * dist[n];
        }
    }
    else if (axis==1)
    {
        for (n=0; n<csize-1; n++) 
        {
            simdata[d + p*dx + s*dt*dx] += (modely[s + indx[n]*rz + indy[n]*ry*rz] * vx + modelz[s + indx[n]*rz + indy[n]*ry*rz] * vy) * dist[n];
        }
    }
    else if (axis==2)
    {
        for (n=0; n<csize-1; n++) 
        {
            simdata[d + p*dx + s*dt*dx] += (modelx[indx[n] + s*rz + indy[n]*ry*rz] * vx + modelz[indx[n] + s*rz + indy[n]*ry*rz] * vy) * dist[n];
        }
    }
}

// void 
// calc_simdata2(
//     int s, int p, int d,
//     int ry, int rz, 
//     int dt, int dx,
//     int csize, const int *indi, const float *dist,
//     float vx, float vy, 
//     const float *modelx, const float *modely, 
//     float *simdata)
// {
//     int n;

//     int index_model = s*ry*rz;
//     int index_data = d+p*dx+s*dt*dx;
//     for (n=0; n<csize-1; n++) 
//     {
//         simdata[index_data] += (modelx[indi[n]+index_model] * vx + modely[indi[n]+index_model] * vy) * dist[n];
//     }
// }
