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

// TODO: Load and save FFTW "wisdom" to file
// TODO: Use guru interface to load real and imag into FFTW without copying
// TODO: Profile code and check adding SIMD to various functions (from OpenMP)

#define USE_FFTW
#define _USE_MATH_DEFINES

#include "gridrec.h"
#ifdef USE_FFTW
    #include <fftw3.h>
#else
    #include "fft.h"
#endif
#include "fft.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
    #define M_PI 3.14159265359
#endif

void 
gridrec(
    const float *data, int dy, int dt, int dx, const float *center, 
    const float *theta, float *recon, int ngridx, int ngridy, const char *fname)
{
    int s, p, iu, iv;
    float *sine, *cose, *wtbl, *work, *winv;

    float (* const filter)(float) = get_filter(fname);
    const float C = 7.0;
    const float nt = 20.0;
    const float lambda = 0.99998546;
    const unsigned int L = (int)(2*C/M_PI);
    const int ltbl = 512;         
    int pdim;
    float _Complex *sino, *filphase, **H;

    const float coefs[11] = {
         0.5767616E+02, -0.8931343E+02,  0.4167596E+02,
        -0.1053599E+02,  0.1662374E+01, -0.1780527E-00,
         0.1372983E-01, -0.7963169E-03,  0.3593372E-04,
        -0.1295941E-05,  0.3817796E-07};
    
    // Compute pdim = next power of 2 >= dx
    for(pdim = 4; pdim < dx; pdim *= 2);

    const int M02 = pdim/2-1;

    // Allocate storage for various arrays.
    sino = malloc_vector_c(pdim);
    filphase = malloc_vector_c(pdim/2);
    H = malloc_matrix_c(pdim, pdim);
    wtbl = malloc_vector_f(ltbl+1);
    winv = malloc_vector_f(pdim-1);
    work = malloc_vector_f(L+1);

    // Set up table of sines and cosines.
    set_trig_tables(dt, theta, &sine, &cose);

    // Set up PSWF lookup tables.
    set_pswf_tables(C, nt, lambda, coefs, ltbl, M02, wtbl, winv);
#ifdef USE_FFTW
    // Set up fftw plans
    fftwf_plan forward_1d;
    fftwf_plan reverse_2d;
    forward_1d = fftwf_plan_dft_1d(pdim, sino, sino, FFTW_FORWARD, FFTW_MEASURE);
    reverse_2d = fftwf_plan_dft_2d(pdim, pdim, H[0], H[0], FFTW_BACKWARD, FFTW_MEASURE);
#endif
    // For each slice.
    for (s=0; s<dy; s+=2)
    {
        // Set up table of combined filter-phase factors.
        set_filter_tables(dt, pdim, center[s], filter, filphase);

        // First clear the array H
        for(iu=0; iu<pdim; iu++) 
        {
            for(iv=0; iv<pdim; iv++)
            {
                H[iu][iv] = 0.0;
            }
        }

        // Loop over the dt projection angles. For each angle, do the following:

        //     1. Copy the real projection data from the two slices into the
        //      real and imaginary parts of the first dx elements of the 
        //      complex array, sino[].  Set the remaining pdim-dx elements
        //      to zero (zero-padding).

        //     2. Carry out a (1D) Fourier transform on the complex data.
        //      This results in transform data that is arranged in 
        //      "wrap-around" order, with non-negative spatial frequencies 
        //      occupying the first half, and negative frequencies the second 
        //      half, of the array, sino[].
            
        //     3. Multiply each element of the 1-D transform by a complex,
        //      frequency dependent factor, filphase[].  These factors were
        //      precomputed as part of recon_init() and combine the 
        //      tomographic filtering with a phase factor which shifts the 
        //      origin in configuration space to the projection of the 
        //      rotation axis as defined by the parameter, "center".  If a 
        //      region of interest (ROI) centered on a different origin has 
        //      been specified [(X0,Y0)!=(0,0)], multiplication by an 
        //      additional phase factor, dependent on angle as well as 
        //      frequency, is required.

        //     4. For each data element, find the Cartesian coordinates, 
        //      <U,V>, of the corresponding point in the 2D frequency plane, 
        //      in  units of the spacing in the MxM rectangular grid placed 
        //      thereon; then calculate the upper and lower limits in each 
        //      coordinate direction of the integer coordinates for the 
        //      grid points contained in an LxL box centered on <U,V>.  
        //      Using a precomputed table of the (1-D) convolving function, 
        //      W, calculate the contribution of this data element to the
        //      (2-D) convolvent (the 2_D convolvent is the product of
        //      1_D convolvents in the X and Y directions) at each of these
        //      grid points, and update the complex 2D array H accordingly.  

        // At the end of Phase 1, the array H[][] contains data arranged in 
        // "natural", rather than wrap-around order -- that is, the origin in 
        // the spatial frequency plane is situated in the middle, rather than 
        // at the beginning, of the array, H[][].  This simplifies the code 
        // for carrying out the convolution (step 4 above), but necessitates 
        // an additional correction -- See Phase 3 below.

        float _Complex Cdata1, Cdata2;
        float U, V, rtmp;
        const float L2 = (int)(C/M_PI);
        const float tblspcg = 2*ltbl/L;

        const int pdim2 = pdim >> 1;
        const int M2 = pdim >> 1;
        int iul, iuh, ivl, ivh;
        int j, k;

        // For each projection
        for(p=0; p<dt; p++)
        {
            for(j=0; j<dx; j++)
            {
                float second_sino = 0.0; //idea: just do same slice twice to process faster
                const unsigned int index = j+p*dx+s*dx*dt;
                if ((s + 1) < dy)
                {
                    second_sino = data[index + dx*dt];
                }
                sino[j] = data[index] + I*second_sino;
            }

            // Zero fill the rest of the array
            for(;j<pdim;j++)
            {
                sino[j] = 0.0;
            }

            // Take FFT of the projection array
#ifdef USE_FFTW
            fftwf_execute(forward_1d);
#else
            four1((float*)sino-1,pdim,1);
#endif
            // For each FFT(projection)
            for(j=1; j<pdim2; j++)
            {    
                Cdata1 = filphase[j] * sino[j];
                Cdata2 = conjf(filphase[j]) * sino[pdim-j];

                U = j * cose[p] + M2;
                V = j * sine[p] + M2;

                // Note freq space origin is at (M2,M2), but we
                // offset the indices U, V, etc. to range from 0 to M-1.
                iul = ceil(U-L2);
                iuh = floor(U+L2);
                ivl = ceil(V-L2); 
                ivh = floor(V+L2);
                if(iul<1) iul = 1;
                if(iuh>=pdim) iuh = pdim-1; 
                if(ivl<1) ivl = 1; 
                if(ivh>=pdim) ivh = pdim-1; 

                // Note aliasing value (at index=0) is forced to zero.
                for(iv=ivl, k=0; iv<=ivh; iv++, k++) {
                    work[k] = wtbl[lroundf(fabs(V-iv)*tblspcg)];
                }
                    
                for(iu=iul; iu<=iuh; iu++)
                {
                    rtmp = wtbl[lroundf(fabs(U-iu)*tblspcg)];
                    for(iv=ivl, k=0; iv<=ivh; iv++, k++)
                    {
                        const float convolv = rtmp*work[k];
                        H[iu][iv] += convolv*Cdata1;
                        H[pdim-iu][pdim-iv] += convolv*Cdata2;
                    }
                }
            }
        }

        // Carry out a 2D inverse FFT on the array H.

        // At the conclusion of this phase, the configuration 
        // space data is arranged in wrap-around order with the origin
        // (center of reconstructed images) situated at the start of the 
        // array.  The first (resp. second) half of the array contains the lower,
        // Y<0 (resp, upper Y>0) part of the image, and within each row of the 
        // array, the first (resp. second) half contains data for the right [X>0]
        // (resp. left [X<0]) half of the image.

#ifdef USE_FFTW
        fftwf_execute(reverse_2d);
#else
        const unsigned long H_size[2] = {pdim, pdim};
        fourn((float*)(*H)-1, H_size-1, 2, -1);
#endif
        // Copy the real and imaginary parts of the complex data from H[][],
        // into the output buffers for the two reconstructed real images, 
        // simultaneously carrying out a final multiplicative correction.  
        // The correction factors are taken from the array, winv[], previously 
        // computed in set_pswf_tables(), and consist logically of three parts, namely:

        //  1. A positive real factor, corresponding to the reciprocal
        //     of the inverse Fourier transform, of the convolving
        //     function, W, and

        //  2. Multiplication by the cell size, (1/D1)^2, in 2D frequency
        //     space.  This correctly normalizes the 2D inverse FFT carried
        //     out in Phase 2.  (Note that all quantities are expressed in
        //     units in which the detector spacing is one.)

        //  3. A sign change for the "odd-numbered" elements (in a 
        //     checkerboard pattern) of the array.  This compensates
        //     for the fact that the 2-D Fourier transform (Phase 2) 
        //     started with a frequency array in which the zero frequency 
        //     point appears in the middle of the array instead of at 
        //     its start.

        // Only the elements in the square M0xM0 subarray of H[][], centered 
        // about the origin, are utilized.  The other elements are not part of the
        // actual region being reconstructed and are discarded.  Because of the 
        // wrap-around ordering, the subarray must actually be taken from the four
        // corners" of the 2D array, H[][] -- See Phase 2 description, above.

        // The final data correponds physically to the linear X-ray absorption
        // coefficient expressed in units of the inverse detector spacing -- to 
        // convert to inverse cm (say), one must divide the data by the detector 
        // spacing in cm.

        int ustart, vstart, ufin, vfin;
        const int padx = (pdim-ngridx)/2;
        const int pady = (pdim-ngridy)/2;
        const int offsetx = M02+1-padx;
        const int offsety = M02+1-pady;
        const int islc1 = s*ngridx*ngridy; // index slice 1
        const int islc2 = (s+1)*ngridx*ngridy; // index slice 2

        ustart = pdim-offsety;
        ufin = pdim;
        j = 0;
        while(j<ngridy)
        {
            for(iu=ustart; iu<ufin; j++, iu++)
            {
                const float corrn_u = winv[j+pady];
                vstart = pdim-offsetx;
                vfin = pdim;
                k = 0;
                while(k<ngridx)
                {
                    for(iv=vstart; iv<vfin; k++, iv++)
                    {
                        const float corrn = corrn_u*winv[k+padx];
                        recon[islc1+ngridy*(ngridx-1-k)+j] = corrn*crealf(H[iu][iv]);
                        if((s+1) < dy)
                        {
                            recon[islc2+ngridy*(ngridx-1-k)+j] = corrn*cimagf(H[iu][iv]);
                        }
                    }
                    if(k<ngridx)
                    {
                        vstart = 0;
                        vfin = ngridx-offsetx;
                    }
                }
            }
            if(j<ngridy) 
            {
                ustart = 0;
                ufin = ngridy-offsety;
            }
        }
    }

    free_vector_f(sine);
    free_vector_f(cose);
    free_vector_c(sino);
    free_vector_f(wtbl);
    free_vector_c(filphase);
    free_vector_f(winv);
    free_vector_f(work);
    free_matrix_c(H);
#ifdef USE_FFTW
    fftwf_destroy_plan(forward_1d);
    fftwf_destroy_plan(reverse_2d);
#endif
    return;
}


void 
set_filter_tables(
    int dt, int pd, float center, 
    float(* const pf)(float), float _Complex *A)
{ 
    // Set up the complex array, filphase[], each element of which
    // consists of a real filter factor [obtained from the function,
    // (*pf)()], multiplying a complex phase factor (derived from the
    // parameter, center}.  See Phase 1 comments.

    const int pd2 = pd/2;
    const float norm = M_PI/pd/dt;
    const float rtmp1 = 2*M_PI*center/pd;
    int j;
    float x, rtmp2;

    for(j=0; j<pd2; j++)
    {
        x = j*rtmp1;
        rtmp2 = (*pf)((float)j/pd)*norm;
        A[j] = rtmp2 * (cosf(x) - I*sinf(x));
    }
}


void 
set_pswf_tables(
    float C, int nt, float lambda, const float *coefs, 
    int ltbl, int linv, float* wtbl, float* winv)                                            
{
    // Set up lookup tables for convolvent (used in Phase 1 of   
    // do_recon()), and for the final correction factor (used in 
    // Phase 3).

    int i;
    float norm;
    const float fac = (float)ltbl / (linv+0.5);
    const float polyz = legendre(nt, coefs, 0.);

    wtbl[0] = 1.0;
    for(i=1; i<=ltbl; i++) 
    {   
        wtbl[i] = legendre(nt, coefs, (float)i/ltbl) / polyz;
    }

    // Note the final result at end of Phase 3 contains the factor, 
    // norm^2.  This incorporates the normalization of the 2D
    // inverse FFT in Phase 2 as well as scale factors involved
    // in the inverse Fourier transform of the convolvent.
    norm = sqrt(M_PI/2/C/lambda) / 1.2;

    winv[linv] = norm / wtbl[0];
    for(i=1; i<=linv; i++)
    {
        // Minus sign for alternate entries
        // corrects for "natural" data layout
        // in array H at end of Phase 1.
        norm = -norm; 
        winv[linv+i] = winv[linv-i] = norm / wtbl[(int)((i*fac)+0.5)];
    }
}


void 
set_trig_tables(int dt, const float *theta, float **sine, float **cose)
{
    // Set up tables of sines and cosines.
    float *s, *c;

    *sine = s = malloc_vector_f(dt);
    *cose = c = malloc_vector_f(dt);

    for(int j=0; j<dt; j++)
    {
        s[j] = sinf(theta[j]);
        c[j] = cosf(theta[j]);
    }
}


float 
legendre(int n, const float *coefs, float x)
{
    // Compute SUM(coefs(k)*P(2*k,x), for k=0,n/2)
    // where P(j,x) is the jth Legendre polynomial.
    // x must be between -1 and 1.
    // FIXME: rename "new", keyword in c++
    float penult, last, cur, y;
    int j, k, even;

    y = coefs[0];
    penult = 1.0;
    last = x;
    even = 1;
    k = 1;
    for(j=2; j<=n; j++)
    {
        cur = (x*(2*j-1)*last-(j-1)*penult)/j;
        if(even)
        {
            y += cur*coefs[k];
            even = 0;
            k++;
        } 
        else
        {
            even = 1;
        }

        penult = last;
        last = cur;
    }
    return y;
}

float*
malloc_vector_f(size_t n) 
{
#ifdef USE_FFTW
    return fftwf_alloc_real(n);
#else
    return (float *) malloc(n * sizeof(float));
#endif
}

void
free_vector_f(float* v)
{
#ifdef USE_FFTW
    fftwf_free(v);
#else
    free(v);
#endif
}

float _Complex*
malloc_vector_c(size_t n) 
{
#ifdef USE_FFTW
    return fftwf_alloc_complex(n);
#else
    return (float _Complex *) malloc(n * sizeof(float _Complex));
#endif
}

void
free_vector_c(float _Complex* v)
{
#ifdef USE_FFTW
    fftwf_free(v);
#else
    free(v);
#endif
}


float _Complex**
malloc_matrix_c(size_t nr, size_t nc)
{
    float _Complex **m = NULL;
    long i;

    // Allocate pointers to rows,
    //m = (float _Complex **) fftwf_malloc(nr * sizeof(float _Complex *));
    m = (float _Complex **) malloc(nr * sizeof(float _Complex *));

    /* Allocate rows and set the pointers to them */
    m[0] = malloc_vector_c(nr * nc);

    for (i = 1; i < nr; i++) 
    {
        m[i] = m[i-1] + nc;
    }
    return m;
}

void
free_matrix_c(float _Complex** m)
{
    free_vector_c(m[0]);
    free(m);
}

// No filter
float 
filter_none(float x)
{
    return 1;
}


// Shepp-Logan filter
float 
filter_shepp(float x)
{
    return fabs(sin(M_PI*x)/M_PI);
}


// Cosine filter 
float 
filter_cosine(float x)
{
    return fabs(x)*(cos(M_PI*x));
}


// Hann filter 
float 
filter_hann(float x)
{
    float cutoff = 0.5;
    return fabs(x)*0.5*(1.+cos(M_PI*x/cutoff));
}


// Hamming filter 
float 
filter_hamming(float x)
{
    float cutoff = 0.5;
    return fabs(x)*(0.54+0.46*cos(M_PI*x/cutoff));
}

// Ramlak filter
float 
filter_ramlak(float x)
{
    return fabs(x);
}

// Parzen filter
float 
filter_parzen(float x)
{
    float cutoff = 0.5;
    return fabs(x)*pow(1-fabs(x)/cutoff, 3);
}

// Butterworth filter
float 
filter_butterworth(float x)
{
    float cutoff = 0.4;
    float order = 8;
    return fabs(x)/(1+pow(x/cutoff, 2*order));
}


float (*get_filter(const char *name))(float) 
{
    struct 
    {
        const char* name; 
        float (* const fp)(float);
    } fltbl[] = {
        {"none", filter_none},
        {"shepp", filter_shepp}, // Default
        {"cosine", filter_cosine},
        {"hann", filter_hann},
        {"hamming", filter_hamming},
        {"ramlak", filter_ramlak},
        {"parzen", filter_parzen},
        {"butterworth", filter_butterworth}};

    for(int i=0; i<sizeof(fltbl); i++)
    {
        if(!strcmp(name, fltbl[i].name))
        {
            return fltbl[i].fp;
        }
    }
    return fltbl[1].fp;   
}
