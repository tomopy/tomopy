/*-----------------------------------------------------------------------------
 * Copyright (c) 2013, UChicago Argonne, LLC
 * See LICENSE file.
 *---------------------------------------------------------------------------*/

#ifndef TOMORECON_H
#define TOMORECON_H

/*---------------------------------------------------------------------------*/

#include "MessageQueue.h"
#include <boost/date_time.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include "grid.h"


/*---------------------------------------------------------------------------*/

/** Structure that is passed from the constructor to the workerTasks in the toDoQueue */
typedef struct {
  int sliceNumber;  /**< Slice number of first slice */
  float center;     /**< Rotation center to use for these slices */
  float *pIn1;      /**< Pointer to first input slice */
  float *pIn2;      /**< Pointer to second input slice.  Can be NULL */
  float *pOut1;     /**< Pointer to first output slice */
  float *pOut2;     /**< Pointer to second output slice. Can be NULL */
} toDoMessage_t;

/** Structure that is passed from the workerTask to the supervisorTask in the doneQueue */
typedef struct {
  int sliceNumber;      /**< Slice number of first slice */
  int numSlices;        /**< Number of slices that we reconstructed. 1 or 2. */
  double sinogramTime;  /**< Time required to compute the sinograms */
  double reconTime;     /**< Time required to reconstruct */
} doneMessage_t;

/** Structure that is passed to the constructor to define the reconstruction 
    NOTE: This structure must match the structure defined in IDL in tomo_params__define.pro! 
    There are fields in this structure that are not used by tomoRecon, but are present because
    they are used by other reconstruction methods */
typedef struct {
  int numPixels;            /**< Number of horizontal pixels in the input data */
  int numProjections;       /**< Number of projection angles in the input data */
  int numSlices;            /**< Maximum number of slices that will be passed to tomoRecon::reconstruct */
  float sinoScale;          /**< Scale factor to multiply sinogram when airPixels=0 */
  float reconScale;         /**< Scale factor to multiple reconstruction */
  int paddedSinogramWidth;  /**< Number of pixels to pad the sinogram to;  must be power of 2 and >= numPixels */
  int airPixels;            /**< Number of pixels of air on each side of sinogram to use for secondary normalization */
  int ringWidth;            /**< Number of pixels in smoothing kernel when doing ring artifact reduction; 0 disables ring artifact reduction */
  int fluorescence;         /**< Set to 1 if the data are fluorescence data and should not have the log taken when computing sinogram */

  int reconMethod;          /**< 0=tomoRecon, 1=Gridrec, 2=Backproject */
  int reconMethodTomoRecon;
  int reconMethodGridrec;
  int reconMethodBackproject;
  
  int numThreads;           /**< Number of workerTask threads to create */
  int slicesPerChunk;       /**< Number of slices to reconstruct per chunk */
  int debug;                /**< Debug output level; 0: only error messages, 1: debugging from tomoRecon, 2: debugging also from grid */
  char debugFileName[256];  /**< Name of file for debugging output;  use 0 length string ("") to send output to stdout */

  // These are gridRec parameters
  int geom;                 /**< 0 if array of angles provided; 1,2 if uniform in half, full circle */ 
  float pswfParam;          /**< PSWF parameter */
  float sampl;              /**< "Oversampling" ratio */
  float MaxPixSiz;          /**< Max pixel size for reconstruction */
  float ROI;                /**< Region of interest (ROI) relative size */
  float X0;                 /**< Offset of ROI from rotation axis in units of center-to-edge distance */
  float Y0;                 /**< Offset of ROI from rotation axis in units of center-to-edge distance */
  int ltbl;                 /**< Number of elements in convolvent lookup tables */
  char fname[16];           /**< Name of filter function */

  // Backproject parameters
  int BP_Method;            /**< 0=Riemann, 1=Radon */
  int BP_MethodRiemann;
  int BP_MethodRadon;
  char BP_filterName[16];
  int BP_filterSize;
  int RiemannInterpolation;
  int RiemannInterpolationNone;
  int RiemannInterpolationBilinear;
  int RiemannInterpolationCubic;
  int RadonInterpolation;
  int RadonInterpolationNone;
  int RadonInterpolationLinear;
} tomoParams_t;

#ifdef __cplusplus

/** Structure that is used to create a worker task.  This is the structure passed to epicsThreadCreate() */
typedef struct {
   class tomoRecon *pTomoRecon; /**< Pointer to the tomoRecon object */
   int taskNum;                 /**< Task number that is passed to tomoRecon::workerTask */
} workerCreateStruct;

/** Class to do tomography reconstruction.
* Creates a supervisorTask that supervises the reconstruction process, and a set of workerTask threads
* that compute the sinograms and do the reconstruction.
* The reconstruction is done with the GridRec code, originally written at Brookhaven National Lab.
* Gridrec was modified to be thread-safe.
* When the class is created it can be used to reconstruct many slices in a single call, and does
* the reconstruction using multiple threads and cores.  The reconstruction function can be called 
* repeatedly to reconstruct more sets of slices.  Once the object is created it is restricted to
* reconstructing with the same set of parameters, with the exception of the rotation center, which
* can be specified on a slice-by-slice basis.  If the reconstruction parameters change (number of X pixels, 
* number of projections, Gridrec parameters, etc.) then the tomoRecon object must be deleted and a
* new one created.
*/
class tomoRecon {

public:
   tomoRecon(tomoParams_t *pTomoParams, float *pAngles);
   virtual ~tomoRecon();
   virtual int reconstruct(int numSlices, float *center, float *pInput, float *pOutput);
   virtual void workerTask(int taskNum);
   virtual void sinogram(float *pIn, float *pOut);
   virtual void poll(int *pReconComplete, int *pSlicesRemaining);
   virtual void logMsg(const char *pFormat, ...);

private:

   tomoParams_t *pTomoParams_;
   int numPixels_;
   int numSlices_;
   int numProjections_;
   int paddedWidth_;
   int numThreads_;
   float *pAngles_;
   float *pInput_;
   float *pOutput_;
   int queueElements_;
   int debug_;
   FILE *debugFile_;
   int reconComplete_;
   int slicesRemaining_;
   int shutDown_;

   MessageQueue toDoMsgQueue;
   MessageQueue doneMsgQueue;
   MessageQueueId toDoQueue_;
   MessageQueueId doneQueue_;
   boost::mutex m_mutex;

};
#endif

/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
