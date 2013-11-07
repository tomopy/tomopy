/* File grid.c 7/7/98 at BNL
   Major re-write Mark Rivers, July 2012.  
      Convert to C++ to allow multi-threading (no static variables).  
      Use native FFTW interface.
*/

#include "grid.h"

/**** Macros and typedefs used in this module only ****/

#define Cmult(A,B,C) {(A).r=(B).r*(C).r-(B).i*(C).i;\
                      (A).i=(B).r*(C).i+(B).i*(C).r;}
/*** A=B*C for complex A, B, C. A must be distinct, and an lvalue */

#ifdef INTERP
#define Cnvlvnt(X) (wtbl[(int)X]+(X-(int)X)*dwtbl[(int)X])
/* Linear interpolation version */
#else
#define Cnvlvnt(X) (wtbl[(int)(X+0.5)])    
/* Nearest nbr version - no interpolation */
#endif

static void trig_su(sg_struct *SG,float **SP,float **CP);

static float legendre(int n,float *coefs, float x);

/** Constructor for the grid object.
* Allocates memory that used internally. <br/>
* Creates lookup tables of sine and cosine functions for efficiency. <br/>
* Creates the FFTW plans for 1-D and 2-D FFTs.  This operation can require several seconds for
* large arrays.  This overhead time only occurs the very first time the sinograms of a particular size 
* are reconstructed by an application because FFTW caches the plans internally. <br>
* NOTE: Because FFTW plan creation is not thread-safe this constructor is also not thread-safe. If multiple
* grid objects are being created then their creation must be protected by a mutex so that the constructor
* is not be called by multiple threads at the same time.  Once the grid object is created then then it is fully
* thread-safe.
* \param[in] GP Pointer to a grid_struct structure that defines the grid reconstruction parameters
* \param[in] SGP Pointer to an sg_struct structure that defines the sinogram parameters
* \param[out] imgsiz The size of the reconstructed images the grid will produce =float[imgsiz, imgsiz]
*/
grid::grid(grid_struct *GP,sg_struct *SGP, long *imgsiz)

     /************************************************************/
     /* Compute various parameters needed for the reconstruction.*/
     /* Allocate storage for various arrays.                     */
     /* Set up various lookup tables.                            */
     /************************************************************/

{
  float C,MaxPixSiz,R,D0,D1;  /* 7/7/98 */
  long itmp;
  
  pswf_struct *pswf;

  n_ang=SGP->n_ang;
  n_det=SGP->n_det;
  previousCenter = -1.;      
        
  sampl=GP->sampl;
  MaxPixSiz=GP->MaxPixSiz;
  R=GP->R;
  X0=GP->X0;
  Y0=GP->Y0;
  filter=GP->filter;
  ltbl=GP->ltbl;
  pswf=GP->pswf;
  C=pswf->C;
  verbose = GP->verbose;
  if (verbose) printf("grid::grid \n"
       "SGP->n_ang=%d\n"
       "SGP->n_det=%d\n"
       "SGP->geom=%d\n"
       "SGP->angles[100]=%f\n"
       "SGP->center=%f\n"
       "GP->sampl=%f\n"
       "GP->MaxPixSiz=%f\n"
       "GP->R=%f\n"
       "GP->X0=%f\n"
       "GP->Y0=%f\n"
       "GP->fname=%s\n"
       "GP->ltbl=%ld\n",
       SGP->n_ang, SGP->n_det, SGP->geom, SGP->angles[100], SGP->center,
       GP->sampl, GP->MaxPixSiz, GP->R, GP->X0, GP->Y0, GP->fname, GP->ltbl);

  /*** Set flag if ROI offset exists **/
  if(X0!=0.||Y0!=0.)flag=1;  
  else flag=0;
        
  /*** Compute pdim = next power of 2 >=n_det */ 
  pdim=1;
  itmp=n_det-1;
  while(itmp)
    {
      pdim<<=1;
      itmp>>=1;
    }

  D0=R*n_det;  /* Size of ROI */  /* 7/7/98 */
  D1=sampl*D0; /* Size of "extended" region. 7/7/98 */

  /*** Compute raster size M for the oversampled 2D array: */
  /*          M = next power of two >= D1/MaxPixSiz */  /*7/7/98 */
  M=1;
  itmp=D1/MaxPixSiz-1;  /* 7/7/98 */
  while(itmp)
    {
      M<<=1;
      itmp>>=1;
    }

  /*** Compute M0 = raster size for the ROI =
       = largest ODD integer <= M/sampl */

  M02=floor(M/2/sampl-0.5);
  M0=2*M02+1;

  /****/

  GP->sampl=sampl=(float)M/M0; /* Adjust value of sampl */
  D1=sampl*D0;                     /*   and of D1 */  /* 7/7/98 */         
  L=2*C*sampl/pi;          /* Size of convolution support square */
  scale=D1/pdim;   /* Converts freqs to grid units*/ /* 7/7/98 */

  /*** Allocate storage for various arrays */

  cproj = (complex *) fftwf_malloc(sizeof(fftwf_complex) * pdim);
  filphase = (complex *) malloc(sizeof(complex) * pdim/2);        
  HData = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex) * M * M);
  H = (complex **) malloc(M * sizeof(complex *));
  H[0] = (complex *) HData;
  for (itmp=1; itmp<M; itmp++) {
    H[itmp] = H[itmp-1] + M;
  }

  wtbl = (float *) malloc(sizeof(float) * (ltbl+1));        
#ifdef INTERP
  dwtbl = (float *) malloc(sizeof(float) * (ltbl+1));        
#endif
  winv = (float *) malloc(sizeof(float) * M0);
  work = (float *) malloc(sizeof(float) * (L+1));

  /*** Set up table of sines and cosines ***/

  trig_su(SGP,&SINE,&COSE);

  /*** Set up PSWF lookup tables */

  pswf_su(pswf,ltbl,M02,wtbl,dwtbl,winv);

  *imgsiz=M0;
  
  backward_1d_plan = fftwf_plan_dft_1d(pdim, (fftwf_complex *)cproj, (fftwf_complex *)cproj, FFTW_BACKWARD, FFTW_MEASURE);
  forward_2d_plan = fftwf_plan_dft_2d(M, M, HData, HData, FFTW_FORWARD, FFTW_MEASURE);

}

/** Destructor for the grid class
* Releases memory allocated by constructor.<br/>
* NOTE: Memory allocated with fftwf_malloc must be freed with fftwf_free or crashes will result!
*/

grid::~grid()
{
  free(SINE);
  free(COSE);
  fftwf_free(cproj);
  free(filphase);
  free(wtbl);
#ifdef INTERP
  free(dwtbl);
#endif
  free(winv);
  free(work);
  free(H);
  fftwf_free(HData);
}



/** Reconstructs two real slice images from their sinograms.
* This uses the "gridding" algorithm applied to complex data.
* \param[in] center The rotation center to be used for these two slices
* \param[in] G1 Array of pointers to the data for each projection of the first sinogram
* \param[in] G2 Array of pointers to the data for each projection of the second sinogram
* \param[out] S1 Array of pointers to pointers to the data for each row of the reconstruction for the first slice
* \param[out] S2 Array of pointers to pointers to the data for each row of the reconstruction for the second slice
*/
void grid::recon(float center, float** G1,float** G2,float*** S1,float*** S2)
{

  if(verbose)printf(
                    "grid::recon(): center=%f, M0=%ld M= %ld pdim=%ld L=%f scale=%f\n",
                    center, M0,M,pdim,L,scale);

  {        /*** First clear the array H ***/
    long iu,iv;
    for(iu=0;iu<M;iu++) 
      for(iv=0;iv<M;iv++)
        H[iu][iv].r=H[iu][iv].i=0.0;
  }

  if(verbose)printf("Start Phase 1\n");

  {        /***Phase 1 ***************************************

            Loop over the n_ang projection angles. For each angle, do
            the following:

            1. Copy the real projection data from the two slices into the
            real and imaginary parts of the first n_det elements of the 
            complex array, cproj[].  Set the remaining pdim-n_det elements
            to zero (zero-padding).

            2. Carry out a (1D) Fourier transform on the complex data.
            This results in transform data that is arranged in 
            "wrap-around" order, with non-negative spatial frequencies 
            occupying the first half, and negative frequencies the second 
            half, of the array, cproj[].
        
            3. Multiply each element of the 1-D transform by a complex,
            frequency dependent factor, filphase[].  These factors were
            precomputed as part of recon_init() and combine the 
            tomographic filtering with a phase factor which shifts the 
            origin in configuration space to the projection of the 
            rotation axis as defined by the parameter, "center".  If a 
            region of interest (ROI) centered on a different origin has 
            been specified [(X0,Y0)!=(0,0)], multiplication by an 
            additional phase factor, dependent on angle as well as 
            frequency, is required.

            4. For each data element, find the Cartesian coordinates, 
            <U,V>, of the corresponding point in the 2D frequency plane, 
            in  units of the spacing in the MxM rectangular grid placed 
            thereon; then calculate the upper and lower limits in each 
            coordinate direction of the integer coordinates for the 
            grid points contained in an LxL box centered on <U,V>.  
            Using a precomputed table of the (1-D) convolving function, 
            W, calculate the contribution of this data element to the
            (2-D) convolvent (the 2_D convolvent is the product of
            1_D convolvents in the X and Y directions) at each of these
            grid points, and update the complex 2D array H accordingly.  


            At the end of Phase 1, the array H[][] contains data arranged in 
            "natural", rather than wrap-around order -- that is, the origin in 
            the spatial frequency plane is situated in the middle, rather than 
            at the beginning, of the array, H[][].  This simplifies the code 
            for carrying out the convolution (step 4 above), but necessitates 
            an additional correction -- See Phase 3 below.
        **********************************************************************/

    complex Cdata1,Cdata2,Ctmp;
    float U,V,rtmp,L2=L/2.;
    float convolv,tblspcg=2*ltbl/L;

    long pdim2=pdim>>1,M2=M>>1,
      iul,iuh,iu,ivl,ivh,iv,n;

    /* Following are to handle offset ROI case */
    float offset=0.;
    complex phfac;

    // If the center has changed need to call filphase_su
    // This is always done on the first slice, previousCenter is set to -1 to begin
    if (center != previousCenter) {
        /*** Set up table of combined filter-phase factors */
        filphase_su(pdim,center,filter,filphase);
        previousCenter = center;
    }          

    for(n=0;n<n_ang;n++)     /*** Start loop on angles */
      {
        int j,k;
        if(flag) offset=(X0*COSE[n]+Y0*SINE[n])*pi;


        j=0;
        while(j<n_det)        
          {     
            cproj[j].r=G1[n][j];
            cproj[j].i=G2[n][j];
            j++;
          }

        while(j<pdim)        /*** Zero fill the rest of array **/
          {
            cproj[j].r=cproj[j].i=0.0;
            j++;
          }

        fftwf_execute(backward_1d_plan);   
 
        for(j=1;j<pdim2;j++)
          {          /* Start loop on transform data */                        

            if(!flag)
              {
                Ctmp.r=filphase[j].r;
                Ctmp.i=filphase[j].i;
              }
            else
              {
                phfac.r = cos(j*offset);
                phfac.i = -sin(j*offset);
                Cmult(Ctmp,filphase[j],phfac);
              }

            Cmult(Cdata1,Ctmp,cproj[j])
              Ctmp.i=-Ctmp.i;
            Cmult(Cdata2,Ctmp,cproj[pdim-j])

              U=(rtmp=scale*j)*COSE[n]+M2; /* X direction*/
            V=rtmp*SINE[n]+M2;              /* Y direction*/        

            /* Note freq space origin is at (M2,M2), but we
               offset the indices U, V, etc. to range from 0 to M-1 */

            iul=ceil(U-L2);iuh=floor(U+L2);
            ivl=ceil(V-L2);ivh=floor(V+L2);
            if(iul<1)iul=1;if(iuh>=M)iuh=M-1; 
            if(ivl<1)ivl=1;if(ivh>=M)ivh=M-1; 

            /* Note aliasing value (at index=0) is forced to zero */        

            for(iv=ivl,k=0;iv<=ivh;iv++,k++)
              work[k]=Cnvlvnt(abs(V-iv)*tblspcg);
            for(iu=iul;iu<=iuh;iu++)
              {
                rtmp=Cnvlvnt(abs(U-iu)*tblspcg);
                for(iv=ivl,k=0;iv<=ivh;iv++,k++)
                  {
                    convolv = rtmp*work[k];
                    H[iu][iv].r += convolv*Cdata1.r;
                    H[iu][iv].i += convolv*Cdata1.i;
                    H[M-iu][M-iv].r += convolv*Cdata2.r;
                    H[M-iu][M-iv].i += convolv*Cdata2.i;
                  }
              }
          } /*** End loop on transform data */
      } /*** End loop on angles */

  }  /*** End phase 1 ************************************************/        


  if(verbose)printf("Start Phase 2\n");

  {        /*** Phase 2 ********************************************

             Carry out a 2D inverse FFT on the array H.

             At the conclusion of this phase, the configuration 
             space data is arranged in wrap-around order with the origin
             (center of reconstructed images) situated at the start of the 
             array.  The first (resp. second) half of the array
             contains the  lower, Y<0 (resp, upper Y>0) part of the
             image, and within each row of the  array, the first
             (resp. second) half contains data for the right [X>0]
             (resp. left [X<0]) half of the image.

        ********************************************************************/

    fftwf_execute(forward_2d_plan);  

  }  /*** End phase 2 ************************************************/

  if(verbose)printf("Start Phase 3\n");

  { /*** Phase 3 ******************************************************

         Copy the real and imaginary parts of the complex data from H[][],
         into the output buffers for the two reconstructed real images, 
         simultaneously carrying out a final multiplicative correction.  
         The correction factors are taken from the array, winv[], previously 
         computed in pswf_su(), and consist logically of three parts, namely:

         1. A positive real factor, corresponding to the reciprocal
         of the inverse Fourier transform, of the convolving
         function, W, and

         2. Multiplication by the cell size, (1/D1)^2, in 2D frequency
         space.  This correctly normalizes the 2D inverse FFT carried
         out in Phase 2.  (Note that all quantities are ewxpressed in
         units in which the detector spacing is one.)

         3. A sign change for the "odd-numbered" elements (in a 
         checkerboard pattern) of the array.  This compensates
         for the fact that the 2-D Fourier transform (Phase 2) 
         started with a frequency array in which the zero frequency 
         point appears in the middle of the array instead of at 
         its start.

         Only the elements in the square M0xM0 subarray of H[][], centered 
         about the origin, are utilized.  The other elements are not
         part of the actual region being reconstructed and are
         discarded.  Because of the  wrap-around ordering, the
         subarray must actually be taken from the four corners" of the
         2D array, H[][] -- See Phase 2 description, above. 
         The final data correponds physically to the linear X-ray absorption
         coefficient expressed in units of the inverse detector spacing -- to 
         convert to inverse cm (say), one must divide the data by the detector 
         spacing in cm.

    *********************************************************************/

    long iu,iv,j,k,ustart,vstart,ufin,vfin;
    float corrn_u,corrn,**T1,**T2;
    T1=*S1; T2=*S2;

    j=0;
    ustart=(M-M02);
    ufin=M;
    while(j<M0)
      {
        for(iu=ustart;iu<ufin;j++,iu++)
          {
            corrn_u=winv[j];
            k=0;
            vstart=(M-M02);
            vfin=M;
            while(k<M0)
              {
                for(iv=vstart;iv<vfin;k++,iv++)
                  {
                    corrn=corrn_u*winv[k]; 
                    T1[j][k]=corrn*H[iu][iv].r;
                    T2[j][k]=corrn*H[iu][iv].i;
                  }
                if(k<M0) (vstart=0,vfin=M02+1);
              }
          }
        if(j<M0) (ustart=0,ufin=M02+1);
      }

  }  /*** End phase 3 *******************************************************/

  return;

}  /*** End do_recon() ***/




/** Sets up the complex array, filphase[].
* Each element of filphase consists of a real filter factor [obtained from the function,
* (*pf)()], multiplying a complex phase factor (derived from the 
* parameter, center).
*/
/*  See Phase 1 comments in do_recon(), above. */
void grid::filphase_su(long pd, float center,
                        float(*pf)(float), complex *A)
{ 
  long j,pd2=pd>>1;
  float x,rtmp1=2*pi*center/pd,rtmp2;
  float norm=pi/pd/n_ang;        /* Normalization factor for
                                   back transform  7/7/98  */

  if (verbose) printf("filphase_su, pd=%ld, center=%f, pf=%p\n", pd, center, pf);
  for(j=0;j<pd2;j++)
    {
      x=j*rtmp1;
      rtmp2=(*pf)((float)j/pd)*norm;        /* 7/7/98 */
      A[j].r=rtmp2*cos(x);
      A[j].i=-rtmp2*sin(x);
    }

}  /*** End filphase_su() ***/

/** Set up lookup tables for convolvent (used in Phase 1 of grid::recon()), 
* and for the final correction factor (used in Phase 3).
*/
void grid::pswf_su(pswf_struct *pswf,long ltbl, long linv, 
                   float* wtbl,float* dwtbl,float* winv)
{
  float C,*coefs, lmbda, polyz,norm,fac;
  long i;
  int nt;

  C=pswf->C;
  nt=pswf->nt;
  coefs=pswf->coefs;                
  lmbda=pswf->lmbda;        
  polyz=legendre(nt,coefs,0.);


  wtbl[0]=1.0;
  for(i=1;i<=ltbl;i++) 
    {        wtbl[i]=legendre(nt,coefs,(float)i/ltbl)/polyz;
#ifdef INTERP
    dwtbl[i]=wtbl[i]-wtbl[i-1];
#endif
    }

  fac=(float)ltbl/(linv+0.5);
  norm=sqrt(pi/2/C/lmbda)/sampl;        /* 7/7/98 */

  /* Note the final result at end of Phase 3 contains the factor, 
     norm^2.  This incorporates the normalization of the 2D
     inverse FFT in Phase 2 as well as scale factors involved
     in the inverse Fourier transform of the convolvent.
     7/7/98                         */

  winv[linv]=norm/Cnvlvnt(0.);
  for(i=1;i<=linv;i++)
    {
      norm=-norm; 
      /* Minus sign for alternate entries
         corrects for "natural" data layout
         in array H at end of Phase 1.  */

      winv[linv+i]=winv[linv-i]=norm/Cnvlvnt(i*fac);        
    }

}   /*** End pswf_su() ***/


/** Set up tables of sines and cosines. */
static void trig_su(sg_struct *SG, float **SP, float **CP)
{
  int j, geom=SG->geom, n_ang=SG->n_ang;
  float *S,*C;

  *SP = S = (float *) malloc(sizeof(float) * n_ang);
  *CP = C = (float *) malloc(sizeof(float) * n_ang);
  switch (geom)
    {
    case 0:
      {
        float theta,degtorad=pi/180,
          *angle=SG->angles;

        for(j=0;j<n_ang;j++)
          {
            theta=degtorad*angle[j];
            S[j]=sin(theta);
            C[j]=cos(theta);
          }
      }
      break;

    case 1:
    case 2:
      {
        float dtheta=geom*pi/n_ang, dcos, dsin;

        dcos=cos(dtheta);
        dsin=sin(dtheta);
        S[0]=0.0; C[0]=1.0;
        for(j=1;j<n_ang;j++)
          {
            S[j]=dcos*S[j-1]+dsin*C[j-1];
            C[j]=dcos*C[j-1]-dsin*S[j-1];
          }
      }
      break;

    default: 
      fprintf(stderr,
              "Illegal value for angle geometry indicator.\n");
      exit(2);
    }  /*** End switch **/

}        /*** End trig_su ***/


/** Compute SUM(coefs(k)*P(2*k,x), for k=0,n/2) where P(j,x) is the jth Legendre polynomial
*/
static float legendre(int n,float *coefs, float x)
{
  float penult,last,fnew,y;
  int j,k,even;
  if(x>1||x<-1)
    {
      fprintf(stderr, 
              "\nInvalid argument to legendre()");
      exit(2);
    }
  y=coefs[0];
  penult=1.;
  last=x;
  even=1;
  k=1;
  for(j=2;j<=n;j++)
    {
      fnew=(x*(2*j-1)*last-(j-1)*penult)/j;
      if(even)
        {
          y+=fnew*coefs[k];
          even=0;
          k++;
        }
      else        even=1;

      penult=last;
      last=fnew;
    }
  return y;

}   /*** End legendre() ***/
