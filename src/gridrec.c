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

// Possible speedups:
//   * Load and save FFTW "wisdom" to file
//   * Use guru interface to load real and imag into FFTW without copying
//   * Profile code and check adding SIMD to various functions (from OpenMP)

//#define WRITE_FILES
#define _USE_MATH_DEFINES

// Use X/Open-7, where posix_memalign is introduced
#define _XOPEN_SOURCE 700
#include "gridrec.h"
#ifdef USE_MKL
    #include "mkl.h"
#else
    #include <fftw3.h>
#endif
#include <math.h>
#include <string.h>
#include <stdlib.h>
#ifndef USE_MKL
#include <pthread.h>
pthread_mutex_t lock;
#endif

#ifndef M_PI
    #define M_PI 3.14159265359
#endif

#define __LIKELY(x) __builtin_expect(!!(x), 1)
#ifdef __INTEL_COMPILER
#define __PRAGMA_SIMD _Pragma("simd assert")
#define __PRAGMA_SIMD_VECREMAINDER _Pragma("simd assert, vecremainder")
#define __PRAGMA_SIMD_VECREMAINDER_VECLEN8 _Pragma("simd assert, vecremainder, vectorlength(8)")
#define __PRAGMA_OMP_SIMD_COLLAPSE _Pragma("omp simd collapse(2)")
#define __PRAGMA_IVDEP _Pragma("ivdep")
#define __ASSSUME_64BYTES_ALIGNED(x) __assume_aligned((x), 64)
#else
#define __PRAGMA_SIMD
#define __PRAGMA_SIMD_VECREMAINDER
#define __PRAGMA_SIMD_VECREMAINDER_VECLEN8
#define __PRAGMA_OMP_SIMD_COLLAPSE
#define __PRAGMA_IVDEP
#define __ASSSUME_64BYTES_ALIGNED(x)
#endif

void 
gridrec(
    const float *data, int dy, int dt, int dx, const float *center, 
    const float *theta, float *recon, int ngridx, int ngridy, const char *fname,
	const float *filter_par)
{
    int s, p, iu, iv;
    int j, jmin, jmax;
    float *sine, *cose, *wtbl, *winv;
   
#ifdef USE_MKL 
    float *work, *work2;
#else
    float _Complex **work, **work2;
#endif

    float (* const filter)(float, int, int, int, const float*) = get_filter(fname);
    const float C = 7.0;
    const float nt = 20.0;
    const float lambda = 0.99998546;
    const unsigned int L = (int)(2*C/M_PI);
    const int ltbl = 512;         
    int pdim;
    float _Complex *sino, *filphase, *filphase_iter, **H;
    float _Complex **U_d, **V_d;
    float *J_z,*P_z;
#ifndef USE_MKL
    pthread_mutex_lock(&lock); // acquire global lock for set-up
#endif

    const float coefs[11] = {
         0.5767616E+02, -0.8931343E+02,  0.4167596E+02,
        -0.1053599E+02,  0.1662374E+01, -0.1780527E-00,
         0.1372983E-01, -0.7963169E-03,  0.3593372E-04,
        -0.1295941E-05,  0.3817796E-07};
    
    // Compute pdim = next power of 2 >= dx
    for(pdim = 16; pdim < dx; pdim *= 2);

    const int pdim2 = pdim >> 1;
    const int M02 = pdim2 - 1;
    const int M2 = pdim2;

    unsigned char filter2d = filter_is_2d(fname);

    // Allocate storage for various arrays.
#ifdef USE_MKL
    sino = malloc_vector_c(pdim);
#else
    sino = malloc_vector_c(pdim*dt);
#endif
    if(!filter2d){
        filphase = malloc_vector_c(pdim2);
        filphase_iter = filphase;
    }else{
        filphase = malloc_vector_c(dt*(pdim2));
    }
    __ASSSUME_64BYTES_ALIGNED(filphase);
    H = malloc_matrix_c(pdim, pdim); __ASSSUME_64BYTES_ALIGNED(H);
    wtbl = malloc_vector_f(ltbl+1);  __ASSSUME_64BYTES_ALIGNED(wtbl);
    winv = malloc_vector_f(pdim-1);  __ASSSUME_64BYTES_ALIGNED(winv);
    J_z = malloc_vector_f(pdim2*dt); __ASSSUME_64BYTES_ALIGNED(J_z);
    P_z = malloc_vector_f(pdim2*dt); __ASSSUME_64BYTES_ALIGNED(P_z);
    U_d = malloc_matrix_c(dt, pdim); __ASSSUME_64BYTES_ALIGNED(U_d);
    V_d = malloc_matrix_c(dt, pdim); __ASSSUME_64BYTES_ALIGNED(V_d);
#ifdef USE_MKL   
    work = malloc_vector_f(L+1);     __ASSSUME_64BYTES_ALIGNED(work);
    work2 = malloc_vector_f(L+1);    __ASSSUME_64BYTES_ALIGNED(work2)
#else
	work = malloc_matrix_c(pdim2*dt, L+1);  __ASSSUME_64BYTES_ALIGNED(work);
    work2 = malloc_matrix_c(pdim2*dt, L+1); __ASSSUME_64BYTES_ALIGNED(work2);
#endif

    // Set up table of sines and cosines.
    set_trig_tables(dt, theta, &sine, &cose); __ASSSUME_64BYTES_ALIGNED(sine); __ASSSUME_64BYTES_ALIGNED(cose);

    // Set up PSWF lookup tables.
    set_pswf_tables(C, nt, lambda, coefs, ltbl, M02, wtbl, winv);

#ifdef USE_MKL
    DFTI_DESCRIPTOR_HANDLE reverse_1d;
    MKL_LONG length_1d = (MKL_LONG) pdim;
    DftiCreateDescriptor(&reverse_1d, DFTI_SINGLE, DFTI_COMPLEX, 1, length_1d);
    DftiSetValue(reverse_1d, DFTI_THREAD_LIMIT, 1); /* FFT should run sequentially to avoid oversubscription */
    DftiCommitDescriptor(reverse_1d);
    DFTI_DESCRIPTOR_HANDLE forward_2d;
    MKL_LONG length_2d[2] = {(MKL_LONG) pdim, (MKL_LONG) pdim};
    DftiCreateDescriptor(&forward_2d, DFTI_SINGLE, DFTI_COMPLEX, 2, length_2d);
    DftiSetValue(forward_2d, DFTI_THREAD_LIMIT, 1); /* FFT should run sequentially to avoid oversubscription */
    DftiCommitDescriptor(forward_2d);
#else
    int n[1] = {pdim};
    // Set up fftw plans
    fftwf_plan reverse_1d;
    fftwf_plan reverse_1d_many;
    fftwf_plan forward_2d;
    reverse_1d = fftwf_plan_dft_1d(pdim, sino, sino, FFTW_BACKWARD, FFTW_MEASURE);
    reverse_1d_many = fftwf_plan_many_dft(1, n, dt, sino, n, 1, pdim, sino, n, 1, pdim, FFTW_BACKWARD, FFTW_MEASURE);
    forward_2d = fftwf_plan_dft_2d(pdim, pdim, H[0], H[0], FFTW_FORWARD, FFTW_MEASURE);
    pthread_mutex_unlock(&lock); // release global lock
#endif

    for(p=0; p<dt; p++)
    {
        for(j=1; j<pdim2; j++)
        {
            U_d[p][j] = j * cose[p] + M2;
            V_d[p][j] = j * sine[p] + M2;
        }
    }

    float U, V;
    int b;
    // Tune block size depending on architecture
    const int bh = 64;
    const int nb = pdim/bh;
    float wl,wh;
    int z=0;
    const float L2 = (int)(C/M_PI);
    const float tblspcg = 2*ltbl/L;
    int iul, iuh, ivl, ivh;
    int k, k2;

#ifndef USE_MKL
    // Calculations below same for all slices so move outside of slice loop 
    // Set up cache blocking to reduce irregular access
    for(b=0; b<nb; b++)
    {
        wl = bh*b;
        wh = bh*(b+1);
            // Limit j to loop over jmin,jmax and reduce if condition overhead
            for(p=0; p<dt; p++)
            {
                if (cose[p]>0) {
                        jmin = ((wl-M2)/cose[p]);
                        jmax = ceil(((wh-M2)/cose[p]));
                    } else {
                        jmin = ((wh-M2)/cose[p]);
                        jmax = ceil(((wl-M2)/cose[p]));
                    }
                if (jmin<1) {
                        jmin = 1;
                }
                if (jmax>pdim2) {
                        jmax=pdim2;
                    }
                for(j=jmin; j<jmax; j++)
                {
                    U = U_d[p][j];
                    if (U >= wl && U < wh)
                    {
                        J_z[z] = j;
                        P_z[z] = p;
                        V = V_d[p][j];

                        iul = ceil(U-L2);
                        iuh = floor(U+L2);
                        ivl = ceil(V-L2);
                        ivh = floor(V+L2);
                        if(iul<1) iul = 1;
                        if(iuh>=pdim) iuh = pdim-1;
                        if(ivl<1) ivl = 1;
                        if(ivh>=pdim) ivh = pdim-1;

                        // Pre-compute work and work2
                        // Note aliasing value (at index=0) is forced to zero.
                        __PRAGMA_SIMD_VECREMAINDER_VECLEN8
                        for(iv=ivl, k=0; iv<=ivh; iv++, k++) {
                            work[z][k] = wtbl[(int) roundf(fabsf(V-iv)*tblspcg)];
                        }
                        __PRAGMA_SIMD_VECREMAINDER_VECLEN8
                        for(iu=iul, k2=0; iu<=iuh; iu++, k2++) {
                            work2[z][k2] = wtbl[(int) roundf(fabsf(U-iu)*tblspcg)];
                        }

                        z++;
                    }

                }
            }
        }
#endif

    // For each slice.
    for (s=0; s<dy; s+=2)
    {
        // Set up table of combined filter-phase factors.
        set_filter_tables(dt, pdim, center[s], filter, filter_par, filphase, filter2d);

        // First clear the array H
        memset(H[0], 0, pdim * pdim * sizeof(H[0][0]));

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
        //      precomputed as part of recofour1((float*)sino-1,pdim,1);n_init() and combine the 
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
        int zlimit = dt*(pdim2-1); 

#ifdef USE_MKL
        // For each projection
        for(p=0; p<dt; p++)
        {
            float sine_p = sine[p], cose_p = cose[p];
            const unsigned int j0 = dx*(p + s*dt), delta_index = dx*dt;

            __PRAGMA_SIMD_VECREMAINDER
            for(j=0; j<dx; j++)
            {
                // Add data from both slices
                float second_sino = 0.0;
                const unsigned int index = j + j0;
                if (__LIKELY((s + 1) < dy))
                {
                    second_sino = data[index + delta_index];
                }
                sino[j] = data[index] + I*second_sino;

            }

            __PRAGMA_SIMD_VECREMAINDER
            for(j=dx; j<pdim; j++) {
                    // Zero fill the rest of the array
                    sino[j] = 0.0;
            }

            DftiComputeBackward(reverse_1d, sino);

            if(filter2d) filphase_iter = filphase + pdim2*p;

            // For each FFT(projection)
            for(j=1; j<pdim2; j++)
            {
                Cdata1 = filphase_iter[j] * sino[j];
                Cdata2 = conjf(filphase_iter[j]) * sino[pdim-j];

                U = j * cose_p + M2;
                V = j * sine_p + M2;

                // Note freq space origin is at (M2,M2), but we
                // offset the indices U, V, etc. to range from 0 to M-1.
                iul = ceilf(U-L2);
                iuh = floorf(U+L2);
                ivl = ceilf(V-L2);
                ivh = floorf(V+L2);
                if(iul<1) iul = 1;
                if(iuh>=pdim) iuh = pdim-1;
                if(ivl<1) ivl = 1;
                if(ivh>=pdim) ivh = pdim-1;

                // Note aliasing value (at index=0) is forced to zero.
                __PRAGMA_SIMD_VECREMAINDER_VECLEN8
                for(iv=ivl, k=0; iv<=ivh; iv++, k++) {
                    work[k] = wtbl[(int) roundf(fabsf(V-iv)*tblspcg)];
                }

                __PRAGMA_SIMD_VECREMAINDER_VECLEN8
                for(iu=iul, k=0; iu<=iuh; iu++, k++) {
                    work2[k] = wtbl[(int) roundf(fabsf(U-iu)*tblspcg)];
                }

        __PRAGMA_OMP_SIMD_COLLAPSE
                for (iu=iul, k2=0; iu <= iuh; iu++, k2++) {
                    for(iv=ivl, k=0; iv<=ivh; iv++, k++) {
                    	const float rtmp = work2[k2];
                        const float convolv = rtmp*work[k];
                        H[iu][iv] += convolv*Cdata1;
                        H[pdim-iu][pdim-iv] += convolv*Cdata2;
                    }
                }
            }
        }

#else
        // For each projection
        for(p=0; p<dt; p++)
        {
            const unsigned int j0 = dx*(p + s*dt), delta_index = dx*dt;

            __PRAGMA_SIMD_VECREMAINDER
            for(j=0; j<dx; j++)
            {
                // Add data from both slices
                float second_sino = 0.0;
                const unsigned int index = j + j0;
                if (__LIKELY((s + 1) < dy))
                {
                    second_sino = data[index + delta_index];
                }
                //sino[j] = data[index] + I*second_sino;
                sino[j+(p*pdim)] = data[index] + I*second_sino;
            }

            __PRAGMA_SIMD_VECREMAINDER
            for(j=dx; j<pdim; j++) {
                    // Zero fill the rest of the array
                    //sino[j] = 0.0;
                    sino[j+(p*pdim)] = 0.0;
            }
        }
            // Take FFT of the projection array
            //fftwf_execute(reverse_1d);
            fftwf_execute(reverse_1d_many);


            // Use re-ordered p,j,U,V from cache-blocking calculations
            // For each FFT(projection)
        	for(z=0; z<zlimit ;z++)
            {
                p = P_z[z];
                j = J_z[z];
                U = U_d[p][j];
                V = V_d[p][j];            

                if(filter2d) filphase_iter = filphase + pdim2*p;
                
                Cdata1 = filphase_iter[j] * sino[j+(p*pdim)];
                Cdata2 = conjf(filphase_iter[j]) * sino[pdim-j+(p*pdim)];

                // Note freq space origin is at (M2,M2), but we
                // offset the indices U, V, etc. to range from 0 to M-1.
                iul = ceilf(U-L2);
                iuh = floorf(U+L2);
                ivl = ceilf(V-L2);
                ivh = floorf(V+L2);
                if(iul<1) iul = 1;
                if(iuh>=pdim) iuh = pdim-1; 
                if(ivl<1) ivl = 1; 
                if(ivh>=pdim) ivh = pdim-1; 

		__PRAGMA_OMP_SIMD_COLLAPSE
                for (iu=iul, k2=0; iu <= iuh; iu++, k2++) {
                    for(iv=ivl, k=0; iv<=ivh; iv++, k++) {
                        const float convolv = work2[z][k2]*work[z][k];                    
                        H[iu][iv] += convolv*Cdata1;
                        H[pdim-iu][pdim-iv] += convolv*Cdata2;
                    }
                }
            }
#endif
        // Carry out a 2D inverse FFT on the array H.

        // At the conclusion of this phase, the configuration 
        // space data is arranged in wrap-around order with the origin
        // (center of reconstructed images) situated at the start of the 
        // array.  The first (resp. second) half of the array contains the lower,
        // Y<0 (resp, upper Y>0) part of the image, and within each row of the
        // array, the first (resp. second) half contains data for the right [X>0]
        // (resp. left [X<0]) half of the image.

#ifdef USE_MKL
	DftiComputeForward(forward_2d, H[0]);
#else
        fftwf_execute(forward_2d);
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

        // The final data corresponds physically to the linear X-ray absorption
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
                    __PRAGMA_SIMD
                    for(iv=vstart; iv<vfin; k++, iv++)
                    {
                        const float corrn = corrn_u*winv[k+padx];
                        recon[islc1+ngridy*(ngridx-1-k)+j] = corrn*crealf(H[iu][iv]);
                        if(__LIKELY((s+1) < dy))
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
#ifdef USE_MKL
    free_vector_f(work);
#else
    free_matrix_c(work);
    free_matrix_c(work2);    
#endif
    free_matrix_c(H);
    free_vector_f(J_z);
    free_vector_f(P_z);
    free_matrix_c(U_d);
    free_matrix_c(V_d);
#ifdef USE_MKL
    DftiFreeDescriptor(&reverse_1d);
    DftiFreeDescriptor(&forward_2d);
#else
    fftwf_destroy_plan(reverse_1d);
    fftwf_destroy_plan(reverse_1d_many);
    fftwf_destroy_plan(forward_2d);
#endif
    return;
}


void 
set_filter_tables(
    int dt, int pd, float center, 
    float(* const pf)(float, int, int, int, const float *), const float *filter_par, float _Complex *A, unsigned char filter2d)
{ 
    // Set up the complex array, filphase[], each element of which
    // consists of a real filter factor [obtained from the function,
    // (*pf)()], multiplying a complex phase factor (derived from the
    // parameter, center}.  See Phase 1 comments.

    const float norm = M_PI/pd/dt;
    const float rtmp1 = 2*M_PI*center/pd;
    int j,i;
    int pd2 = pd/2;
    float x;
    
    if(!filter2d){
        for(j=0; j<pd2; j++)
        {
            A[j] = (*pf)((float)j/pd, j, 0, pd2, filter_par);
        }

        __PRAGMA_SIMD
        for(j=0; j<pd2; j++)
        {
            x = j*rtmp1;
            A[j] *= (cosf(x) - I*sinf(x))*norm;
        }
    }else{
        for(i=0;i<dt;i++)        
        {
            int j0 = i * pd2;

            for(j=0; j<pd2; j++)
            {
                A[j0 + j] = (*pf)((float)j/pd, j, i, pd2, filter_par);
            }

            __PRAGMA_SIMD
            for(j=0; j<pd2; j++)
            {
                x = j*rtmp1;
                A[j0 + j] *= (cosf(x) - I*sinf(x))*norm;
            }
        }
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
    __PRAGMA_IVDEP
    for(i=1; i<=linv; i++)
    {
        // Minus sign for alternate entries
        // corrects for "natural" data layout
        // in array H at end of Phase 1.
        norm = -norm; 
        winv[linv+i] = winv[linv-i] = norm / wtbl[(int) roundf(i*fac)];
    }
}


void 
set_trig_tables(int dt, const float *theta, float **sine, float **cose)
{
    // Set up tables of sines and cosines.
    float *s, *c;

    *sine = s = malloc_vector_f(dt); __ASSSUME_64BYTES_ALIGNED(s);
    *cose = c = malloc_vector_f(dt); __ASSSUME_64BYTES_ALIGNED(c);

    __PRAGMA_SIMD
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
    float penult, last, cur, y, mxlast;

    y = coefs[0];
    penult = 1.0;
    last = x;
    for(int j=2; j<=n; j++)
    {
	mxlast = -(x*last);
	cur = -(2*mxlast + penult) + (penult + mxlast)/j;
        // cur = (x*(2*j-1)*last-(j-1)*penult)/j;
        if(!(j&1)) // if j is even
        {
            y += cur*coefs[j >> 1];
        } 

        penult = last;
        last = cur;
    }
    return y;
}

static inline void* 
malloc_64bytes_aligned(size_t sz)
{
    #ifdef __MINGW32__
    return  __mingw_aligned_malloc(sz, 64);  
    #else
    void *r = NULL;
    int err = posix_memalign(&r, 64, sz);
    return (err) ? NULL : r;
    #endif
}

inline float*
malloc_vector_f(size_t n) 
{
#ifdef USE_MKL
    return (float *)malloc(n*sizeof(float));
#else
    return fftwf_alloc_real(n);
#endif
}

inline void
free_vector_f(float* v)
{
#ifdef USE_MKL
    free(v);
#else
    fftwf_free(v);
#endif
}

inline float _Complex*
malloc_vector_c(size_t n) 
{
#ifdef USE_MKL
    return (float _Complex*)malloc(n*sizeof(float _Complex));
#else
    return fftwf_alloc_complex(n);
#endif
}

inline void
free_vector_c(float _Complex* v)
{
#ifdef USE_MKL
    free(v);
#else
    fftwf_free(v);
#endif
}


float _Complex**
malloc_matrix_c(size_t nr, size_t nc)
{
    float _Complex **m = NULL;
    size_t i;

    // Allocate pointers to rows,
    m = (float _Complex **) malloc_64bytes_aligned(nr * sizeof(float _Complex *));

    /* Allocate rows and set the pointers to them */
    m[0] = malloc_vector_c(nr * nc);

    for (i = 1; i < nr; i++) 
    {
        m[i] = m[i-1] + nc;
    }
    return m;
}

inline void
free_matrix_c(float _Complex** m)
{

    free_vector_c(m[0]);
    #ifdef __MINGW32__
        __mingw_aligned_free (m);
    #else
        free(m);
    #endif
}

// No filter
float
filter_none(float x, int i, int j, int fwidth, const float* pars)
{
    return 1.0;
}


// Shepp-Logan filter
float
filter_shepp(float x, int i, int j, int fwidth, const float* pars)
{
    if(i==0) return 0.0;
    return fabsf(2*x)*(sinf(M_PI*x)/(M_PI*x));
}


// Cosine filter 
float
filter_cosine(float x, int i, int j, int fwidth, const float* pars)
{
    return fabsf(2*x)*(cosf(M_PI*x));
}


// Hann filter 
float
filter_hann(float x, int i, int j, int fwidth, const float* pars)
{
    return fabsf(2*x)*0.5*(1.+cosf(2*M_PI*x/pars[0]));
}


// Hamming filter 
float
filter_hamming(float x, int i, int j, int fwidth, const float* pars)
{
    return fabsf(2*x)*(0.54+0.46*cosf(2*M_PI*x/pars[0]));
}

// Ramlak filter
float
filter_ramlak(float x, int i, int j, int fwidth, const float* pars)
{
    return fabsf(2*x);
}

// Parzen filter
float
filter_parzen(float x, int i, int j, int fwidth, const float* pars)
{
    return fabsf(2*x)*pow(1-fabs(x)/pars[0], 3);
}

// Butterworth filter
float
filter_butterworth(float x, int i, int j, int fwidth, const float* pars)
{
    return fabsf(2*x)/(1+pow(x/pars[0], 2*pars[1]));
}

// Custom filter
float
filter_custom(float x, int i, int j, int fwidth, const float* pars)
{
    return pars[i];
}

// Custom 2D filter
float
filter_custom2d(float x, int i, int j, int fwidth, const float* pars)
{
    return pars[j*fwidth+i];
}


float (*get_filter(const char *name))(float, int, int, int, const float*) 
{
    struct 
    {
        const char* name; 
        float (* const fp)(float, int, int, int, const float*);
    } fltbl[] = {
        {"none", filter_none},
        {"shepp", filter_shepp}, // Default
        {"cosine", filter_cosine},
        {"hann", filter_hann},
        {"hamming", filter_hamming},
        {"ramlak", filter_ramlak},
        {"parzen", filter_parzen},
        {"butterworth", filter_butterworth},
        {"custom", filter_custom},
        {"custom2d", filter_custom2d}
    };

    for(int i=0; i<10; i++)
    {
        if(!strncmp(name, fltbl[i].name, 16))
        {
            return fltbl[i].fp;
        }
    }
    return fltbl[1].fp;   
}

unsigned char
filter_is_2d(const char *name)
{
    if(!strncmp(name, "custom2d", 16)) return 1;
    return 0;
}
