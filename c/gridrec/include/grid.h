/*** File grid.h  12:27 PM 11/7/97 **/

#define ANSI

/**** System includes */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <time.h>
#include <sys/stat.h>

#include <fftw3.h>

/**** Macros and typedefs ****/
#ifndef max
#define max(A,B) ((A)>(B)?(A):(B))
#endif
#ifndef min
#define min(A,B) ((A)<(B)?(A):(B))
#endif
#define free_matrix(A) (free(*(A)),free(A))
#define abs(A) ((A)>0 ?(A):-(A))
#define pi 3.14159265359
#define TOLERANCE 0.1	/* For comparing centers of two sinograms */
#define LTBL_DEF 512	/* Default lookup table length */

/** Structure used to hold complex data as 2 floats in grid */
typedef struct {
    float r;  /**< Real part */
    float i;  /**< Imaginary part */
} complex;

/** Structure used to define sinograms for grid */
typedef struct {
   int n_ang;	    /**< Number of angles in sinogram */
   int n_det;	    /**< Number of pixels (detectors) per angle */
   int geom;		  /**< 0 if array of angles provided; 1,2 if uniform in half,full circle */ 
   float *angles;	/**< Pointer to the array of angles, if used */
   float center;	/**< Rotation axis location */
} sg_struct;

/** Prolate spheroidal wave function (PSWF) data */
typedef struct {
   float C;	        /**< Parameter for particular 0th order pswf being used */
   int nt;          /**< Degree of Legendre polynomial expansion */
   float lmbda; 	  /**< Eigenvalue */
   float coefs[15];	/**< Coefficients for Legendre polynomial expansion */
} pswf_struct;

/** Parameters for gridding algorithm */
typedef struct {
   pswf_struct *pswf;	     /**< Pointer to data for PSWF being used  */
   float sampl;	  	       /**< "Oversampling" ratio */
   float MaxPixSiz; 	     /**< Maximum pixel size for reconstruction */
   float R;		             /**< Region of interest (ROI) relative size */
   float X0;		           /**< Offset of ROI from rotation axis, in units of center-to-edge distance. */
   float Y0;		           /**< Offset of ROI from rotation axis, in units of center-to-edge distance. */
   char fname[16];		     /**< Name of filter function   */		
   float (*filter)(float); /**< Pointer to filter function */
   long ltbl;		           /**< Number of elements in convolvent lookup tables. */
   int verbose;            /**< Debug printing flag */
} grid_struct;

#ifdef __cplusplus

/** Class to reconstruct parallel beam tomography data using the Gridrec FFT code.
* This code was originally written by Bob Marr and Graham Campbell from
* Brookhaven National Laboratory.  Unfortunately they never published a paper
* on it, so there is no good reference.<br/>
* The original version was written in C and used static C variables to share information
* between functions. It was thus not thread safe.  This version is converted to C++, and
* all variables that are shared between functions are private member data. It is thus thread
* safe as long as each thread creates its own grid object. <br/>
* The original version used the Numerical Recipes FFT functions four1 and fourn. I had previously
* written wrapper routines that maintained the Numerical Recipes API, but used the FFTW 
* library for faster FFTs.  Those wrapper routines were also not thread-safe, and they copied
* data, so were somewhat inefficient.  This version of Gridrec has been changed to use the FFTW API
* directly, it no longer uses the Numerical Recipes API. */
class grid {  
public:
  grid(grid_struct *GP,sg_struct *SGP, long *imgsiz);
  ~grid();
  void recon(float center, float** G1,float** G2,float*** S1,float*** S2);
  void filphase_su(long pd,float fac, float(*pf)(float),complex *A);
  void pswf_su(pswf_struct *pswf,long ltbl, 
               long linv, float* wtbl,float* dwtbl,float* winv);
  
private:
  int flag;       
  int n_det;
  int n_ang;
  long pdim;
  long M;
  long M0;
  long M02;
  long ltbl;
  float sampl;
  float scale;
  float L;
  float X0;
  float Y0;
  float *SINE;
  float *COSE;
  float *wtbl; 
  float *dwtbl;
  float *work;
  float *winv;
  float previousCenter;
  float (*filter)(float);
  complex *cproj;
  complex *filphase;
  complex **H;
  fftwf_complex *HData;
  
  fftwf_plan backward_1d_plan;
  fftwf_plan forward_2d_plan;
  int verbose;   /* Debug printing flag */
};
#endif

/** Global variables **/

#ifdef __cplusplus
extern "C" {
#endif

/**** Function Prototypes ****/

/** Defined in pswf.c **/
void get_pswf(float C, pswf_struct **P);

/** Defined in filters.c  **/
float (*get_filter(char *name))(float);

#ifdef __cplusplus
}
#endif
