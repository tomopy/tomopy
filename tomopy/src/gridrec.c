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

#include "gridrec.h"


static int dz, dx;
static long pdim, M, M0, M02, ltbl;
static float L;
static float *SINE, *COSE, *wtbl, *dwtbl, *work, *winv;
static complex *sino, *filphase, **H;


static float *malloc_vector_f (long n);
static complex *malloc_vector_c (long n);
static complex **malloc_matrix_c (long nr, long nc);

static void 
set_filter_tables(
    long pd, float fac, float(*pf)(float), complex *A);

static void 
set_trig_tables(
    data_pars *SG,float **SP, float **CP);

static void 
set_pswf_tables(
    pswf_struct *pswf, long ltbl, long linv, 
    float* wtbl, float* dwtbl, float* winv);

static float 
legendre(
    int n, float *coefs, float x);



void 
gridrec_init(data_pars *dpars, grid_pars *gpars)
{
    float center;
    float (*filter)(float);
    long itmp;

    printf("dx=%d, dy=%d, dz=%d \n", dpars->dx, dpars->dy, dpars->dz);

    
    pswf_struct *pswf;

    dx = dpars->dx;
    dz = dpars->dz;
    center = dpars->center; 
    
    filter = gpars->filter;
    ltbl = LTBL_DEF;
    pswf = gpars->pswf;
    
    // Compute pdim = next power of 2 >=dz
    pdim = 1;
    itmp = dz - 1;
    while(itmp)
    {
        pdim <<= 1;
        itmp >>= 1;
    }

    M = pdim;
    M0 = pdim-1;
    M02 = pdim/2-1;
    L = (int)2*pswf->C/PI;

    printf("M=%lu, M0=%lu, M02=%lu, pdim=%lu, L=%f \n", M, M0, M02, pdim, L);

    // Allocate storage for various arrays.
    sino = malloc_vector_c(pdim); 
    filphase = malloc_vector_c(pdim / 2);   
    H = malloc_matrix_c(pdim, pdim);
    wtbl = malloc_vector_f(ltbl + 1);
    winv = malloc_vector_f(pdim - 1);
    work = malloc_vector_f((int)2*pswf->C/PI + 1);

    // Set up table of sines and cosines.
    set_trig_tables(dpars, &SINE, &COSE);

    // Set up table of combined filter-phase factors.
    set_filter_tables(pdim, center, filter, filphase);           

    // Set up PSWF lookup tables.
    set_pswf_tables(pswf, ltbl, M02, wtbl, dwtbl, winv);
}



void 
gridrec_main(
    float* data, data_pars *dpars, 
    float* recon, recon_pars *rpars)
{
    grid_pars gpars;

    get_pswf(4.0, &gpars.pswf);
    gpars.filter = get_filter(gpars.fname);

    // printf("dx=%d, dy=%d, dz=%d \n", dpars->dx, dpars->dy, dpars->dz);
    
    gridrec(data, dpars, recon, &gpars);
}


void 
gridrec(
    float* data, data_pars *dpars, 
    float* recon, grid_pars *gpars)
{  
    printf("dx=%d, dy=%d, dz=%d \n", dpars->dx, dpars->dy, dpars->dz);

    float ***data3d = convert(data, dpars->dx, dpars->dy, dpars->dz);
    float ***recon3d = convert(recon, dpars->dy, 2048, 2048);

    gridrec_init(dpars, gpars);

    // First clear the array H
    for(long u=0; u<M; u++) 
    {
        for(long v=0; v<M; v++)
        {
            H[u][v].r = H[u][v].i = 0.0;
        }
    }

    // Loop over the dx projection angles. For each angle, do the following:

    //     1. Copy the real projection data from the two slices into the
    //      real and imaginary parts of the first dz elements of the 
    //      complex array, sino[].  Set the remaining pdim-dz elements
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

    complex Cdata1, Cdata2, Ctmp;
    float U, V, rtmp, L2 = (int)gpars->pswf->C/PI;
    float convolv, tblspcg = 2*ltbl/L;
    printf("L=%f \n", L);

    long pdim2 = pdim >> 1, M2 = M >> 1;
    long iul, iuh, iu, ivl, ivh, iv, n;

    // For each projection
    for(n=0; n<dx; n++)
    {
        int j, k;

        j = 0;
        while(j<dz)  
        {     
            sino[j].r = data3d[n][0][j];
            sino[j].i = data3d[n][1][j];
            // printf("data[%d][%d]=%f \n", n, j, data3d[n][0][j]);
            j++;
        }

        // Zero fill the rest of the array
        while(j<pdim)
        {
            sino[j].r = sino[j].i = 0.0;
            j++;
        }

        // Take FFT of the projection array
        four1((float*)sino-1,pdim,1); 

        // For each FFT(projection)
        for(j=1; j<pdim2; j++)
        {    
            Ctmp.r = filphase[j].r;
            Ctmp.i = filphase[j].i;

            Cmult(Cdata1, Ctmp, sino[j])
            Ctmp.i = -Ctmp.i;
            Cmult(Cdata2, Ctmp, sino[pdim-j])

            U = (rtmp=j) * COSE[n] + M2;
            V = rtmp * SINE[n] + M2;

            // Note freq space origin is at (M2,M2), but we
            // offset the indices U, V, etc. to range from 0 to M-1.
            iul = ceil(U-L2); iuh=floor(U+L2);
            ivl = ceil(V-L2); ivh=floor(V+L2);
            if(iul<1)iul = 1; if(iuh>=M)iuh = M-1; 
            if(ivl<1)ivl = 1; if(ivh>=M)ivh = M-1; 

            // Note aliasing value (at index=0) is forced to zero.
            for(iv=ivl, k=0; iv<=ivh; iv++, k++) {
                work[k] = Cnvlvnt(abs(V-iv)*tblspcg);
            }
                
            for(iu=iul; iu<=iuh; iu++)
            {
                rtmp = Cnvlvnt(abs(U-iu)*tblspcg);
                for(iv=ivl, k=0; iv<=ivh; iv++, k++)
                {
                    convolv = rtmp*work[k];
                    H[iu][iv].r += convolv*Cdata1.r;
                    H[iu][iv].i += convolv*Cdata1.i;
                    H[M-iu][M-iv].r += convolv*Cdata2.r;
                    H[M-iu][M-iv].i += convolv*Cdata2.i;
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

    unsigned long H_size[2];
    H_size[0] = H_size[1] = M;
    fourn((float*)(*H)-1, H_size-1, 2, -1);  

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
    //     out in Phase 2.  (Note that all quantities are ewxpressed in
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

    long j, k, ustart, vstart, ufin, vfin;
    float corrn_u, corrn;

    j = 0;
    ustart = M-M02;
    ufin = M;
    while(j<M0)
    {
        for(iu=ustart; iu<ufin; j++, iu++)
        {
            corrn_u = winv[j];
            k = 0;
            vstart = (M-M02);
            vfin = M;
            while(k<M0)
            {
                for(iv=vstart; iv<vfin; k++, iv++)
                {
                    corrn = corrn_u*winv[k]; 
                    recon3d[0][j][k] = corrn*H[iu][iv].r;
                    recon3d[1][j][k] = corrn*H[iu][iv].i;
                }
                if(k<M0)
                {
                    vstart = 0;
                    vfin = M02+1;
                }
            }
        }
        if(j<M0) 
        {
            ustart = 0;
            ufin = M02+1;
        }
    }

    gridrec_free();
    return;
}


void 
gridrec_free(void)
{
    free(SINE);
    free(COSE);
    free(sino);
    free(filphase);
    free(wtbl);
    free(winv);
    free(work);
    free_matrix(H);
}


float*** 
convert(float *arr, int dim0, int dim1, int dim2)
{
    // Converts a 1-D array to 3-D array given dimensions.
    float ***r3;
    r3 = (float ***) malloc(dim0 * sizeof(float**));

    for(int i=0; i<dim0; i++)
    {
        r3[i] = (float **) malloc(dim1 * sizeof(float*));
    }
        
    for(int i=0; i<dim0; i++) 
    {
        for(int j=0; j<dim1; j++) 
        {
            r3[i][j] = arr + i*dim1*dim2 + j*dim2;
        }
    }
    return r3;
}


static void 
set_filter_tables(
    long pd, float center, 
    float(*pf)(float), complex *A)
{ 
    // Set up the complex array, filphase[], each element of which
    // consists of a real filter factor [obtained from the function,
    // (*pf)()], multiplying a complex phase factor (derived from the
    // parameter, center}.  See Phase 1 comments in do_recon(), above.

    long j, pd2 = pd>>1;
    float x, rtmp1 = 2*PI*center/pd, rtmp2;
    float norm = PI/pd/dx;

    for(j=0; j<pd2; j++)
    {
        x = j*rtmp1;
        rtmp2 = (*pf)((float)j/pd)*norm;
        A[j].r = rtmp2*cosf(x);
        A[j].i = -rtmp2*sinf(x);
    }
}


static void 
set_pswf_tables(
    pswf_struct *pswf,long ltbl, long linv, 
    float* wtbl,float* dwtbl,float* winv)                                            
{
    // Set up lookup tables for convolvent (used in Phase 1 of   
    // do_recon()), and for the final correction factor (used in 
    // Phase 3).

    float C, *coefs, lmbda, polyz, norm, fac;
    long i;
    int nt;

    C = pswf->C;
    nt = pswf->nt;
    coefs = pswf->coefs;      
    lmbda = pswf->lmbda;  
    polyz = legendre(nt, coefs, 0.);

    wtbl[0] = 1.0;
    for(i=1; i<=ltbl; i++) 
    {   wtbl[i] = legendre(nt, coefs, (float)i/ltbl) / polyz;
    }

    fac = (float)ltbl / (linv+0.5);

    // Note the final result at end of Phase 3 contains the factor, 
    // norm^2.  This incorporates the normalization of the 2D
    // inverse FFT in Phase 2 as well as scale factors involved
    // in the inverse Fourier transform of the convolvent.
    norm = sqrt(PI/2/C/lmbda) / 1.2;

    winv[linv] = norm / Cnvlvnt(0.);
    for(i=1; i<=linv; i++)
    {
        // Minus sign for alternate entries
        // corrects for "natural" data layout
        // in array H at end of Phase 1.
        norm = -norm; 
        winv[linv+i] = winv[linv-i] = norm / Cnvlvnt(i*fac);  
    }
}


static void 
set_trig_tables(data_pars *SG, float **SP, float **CP)
{
    // Set up tables of sines and cosines.
    int j, dx = SG->dx;
    float *S, *C;

    *SP = S = malloc_vector_f(dx);
    *CP = C = malloc_vector_f(dx);

    for(j=0; j<dx; j++)
    {
        S[j] = sinf(SG->proj_angle[j]);
        C[j] = cosf(SG->proj_angle[j]);
    }
}


static float 
legendre(int n,float *coefs, float x)
{
    // Compute SUM(coefs(k)*P(2*k,x), for k=0,n/2)
    // where P(j,x) is the jth Legendre polynomial.
    // x must be between -1 and 1.
    float penult, last, new, y;
    int j, k, even;

    y = coefs[0];
    penult = 1.;
    last = x;
    even = 1;
    k = 1;
    for(j=2; j<=n; j++)
    {
        new = (x*(2*j-1)*last-(j-1)*penult)/j;
        if(even)
        {
            y += new*coefs[k];
            even = 0;
            k++;
        } 
        else
        {
            even=1;
        }

        penult = last;
        last = new;
    }
    return y;
}


float*
malloc_vector_f(long n) 
{
    float *v = NULL;
    v = (float *) malloc((size_t) (n * sizeof(float)));
    return v;
}


complex*
malloc_vector_c(long n) 
{
    complex *v = NULL;
    v = (complex *) malloc((size_t) (n * sizeof(complex)));
    return v;
}


complex**
malloc_matrix_c(long nr, long nc)
{
    complex **m = NULL;
    long i;

    /* Allocate pointers to rows */
    m = (complex **) malloc((size_t) (nr * sizeof(complex *)));

    /* Allocate rows and set the pointers to them */
    m[0] = (complex *) malloc((size_t) (nr * nc * sizeof(complex)));

    for (i = 1; i < nr; i++) 
    {
        m[i] = m[i-1] + nc;
    }
    return m;
}


// Shepp-Logan filter
float 
shlo(float x)
{
    return abs(sin(PI*x)/PI);
}


// Hann filter 
float 
hann(float x)
{
    return abs(x)*0.5*(1.+cos(2*PI*x));
}


// Hamming filter 
float 
hamm(float x)
{
    return abs(x)*(0.54+0.46*cos(2*PI*x));
}

// Ramp filter
float 
ramp(float x)
{
    return abs(x);
}

struct 
{
    char* name; 
    float (*fp)(float);
} fltbl[] = {
    {"shlo", shlo}, // Default
    {"shepp", shlo},
    {"hann", hann},
    {"hamm", hamm},
    {"hamming", hamm},
    {"ramp", ramp},
    {"ramlak", ramp}};


float (*get_filter(char *name))(float) 
{
    for(int i=0; i<LEN_FTBL; i++)
    {
        if(!strcmp(name, fltbl[i].name))
        {
            return fltbl[i].fp;
        }
    }
    return fltbl[0].fp;   
}

#define NO_PSWFS 5
pswf_struct pswf_db[NO_PSWFS] = {
    {4.0, 16, 0.99588549,
        {0.5239891E+01, -0.5308499E+01,  0.1184591E+01, 
        -0.1230763E-00,  0.7371623E-02, -0.2864074E-03,
         0.7789983E-05, -0.1564700E-06,  0.2414647E-08}},
    {4.2, 16, 0.99657887,
        {0.6062942E+01, -0.6450252E+01,  0.1551875E+01,
        -0.1755960E-01,  0.1150712E-01, -0.4903653E-03,
         0.1464986E-04, -0.3235110E-06,  0.5492141E-08}},
    {5.0, 18, 0.99935241,
        {0.1115509E+02, -0.1384861E+02,  0.4289811E+01,
        -0.6514303E-00,  0.5844993E-01, -0.3447736E-02,
         0.1435066E-03, -0.4433680E-05,  0.1056040E-06,
        -0.1997173E-08}},
    {6.0, 18, 0.9990188,
        {0.2495593E+02, -0.3531124E+02,  0.1383722E+02,
        -0.2799028E+01,  0.3437217E-00, -0.2818024E-01,
         0.1645842E-02, -0.7179160E-04,  0.2424510E-05,
        -0.6520875E-07}},
    {7.0, 20, 0.99998546,
        {0.5767616E+02, -0.8931343E+02,  0.4167596E+02,
        -0.1053599E+02,  0.1662374E+01, -0.1780527E-00,
         0.1372983E-01, -0.7963169E-03,  0.3593372E-04,
        -0.1295941E-05,  0.3817796E-07}}
};

void get_pswf(float C, pswf_struct **P)
{
    int i = 0;
    while(i<NO_PSWFS && abs(C-pswf_db[i].C)>0.01) 
    {
        i++;
    }
    *P = &pswf_db[i];
    return;
}