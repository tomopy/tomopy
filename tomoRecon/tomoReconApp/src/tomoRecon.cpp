/*
 * tomoRecon.cpp
 * 
 * C++ class for doing computed tomographdeby reconstruction using Gridrec.
 *
 * This runs the reconstruction using multiple threads, each thread reconstructing a set of slices.
 *
 * It uses the EPICS libCom library for OS-independent functions for threads, mutexes, message queues, etc.
 *
 * Author: Mark Rivers
 *
 * Created: July 9, 2012
 */
#include <stdlib.h>
#include <math.h>
#include <stdarg.h>

#include <epicsTime.h>

#include "tomoRecon.h"

extern "C" {
static void supervisorTask(void *pPvt)
{
  tomoRecon *pTomoRecon = (tomoRecon *)pPvt;
  pTomoRecon->supervisorTask();
}

static void workerTask(workerCreateStruct *pWCS)
{
  pWCS->pTomoRecon->workerTask(pWCS->taskNum);
  free(pWCS);
}
} // extern "C"


/** Constructor for the tomoRecon class.
* Creates the message queues for passing messages to and from the workerTask threads.
* Creates the thread that execute the supervisorTask function, 
* and numThreads threads that execute the workerTask function.
* \param[in] pTomoParams A structure containing the tomography reconstruction parameters
* \param[in] pAngles Array of projection angles in degrees */
tomoRecon::tomoRecon(tomoParams_t *pTomoParams, float *pAngles)
  : pTomoParams_(pTomoParams),
    numPixels_(pTomoParams_->numPixels),
    numSlices_(pTomoParams_->numSlices),
    numProjections_(pTomoParams_->numProjections),
    paddedWidth_(pTomoParams_->paddedSinogramWidth),
    numThreads_(pTomoParams_->numThreads),
    pAngles_(pAngles),
    queueElements_((numSlices_+1)/2),
    debug_(pTomoParams_->debug),
    reconComplete_(1),
    shutDown_(0)

{
  epicsThreadId supervisorTaskId;
  char workerTaskName[20];
  epicsThreadId workerTaskId;
  workerCreateStruct *pWCS;
  char *debugFileName = pTomoParams_->debugFileName;
  int i;
  static const char *functionName="tomoRecon::tomoRecon";

  debugFile_ = stdout;
  if ((debugFileName) && (strlen(debugFileName) > 0)) {
    debugFile_ = fopen(debugFileName, "w");
  }
  
  if (debug_) logMsg("%s: entry, creating message queues, events, threads, etc.", functionName);
 
  toDoQueue_ = epicsMessageQueueCreate(queueElements_, sizeof(toDoMessage_t));
  doneQueue_ = epicsMessageQueueCreate(queueElements_, sizeof(doneMessage_t));
  workerWakeEvents_ = (epicsEventId *) malloc(numThreads_ * sizeof(epicsEventId));
  workerDoneEvents_ = (epicsEventId *) malloc(numThreads_ * sizeof(epicsEventId));
  supervisorWakeEvent_ = epicsEventCreate(epicsEventEmpty);
  supervisorDoneEvent_ = epicsEventCreate(epicsEventEmpty);
  fftwMutex_ = epicsMutexCreate();

  /* Create the thread for the supervisor task */
  supervisorTaskId = epicsThreadCreate("supervisorTask",
                                epicsThreadPriorityMedium,
                                epicsThreadGetStackSize(epicsThreadStackMedium),
                                (EPICSTHREADFUNC) ::supervisorTask,
                                this);
  if (supervisorTaskId == 0) {
    logMsg("%s: epicsThreadCreate failure for supervisorTask", functionName); 
  }

  /* Create the worker tasks */
  for (i=0; i<numThreads_; i++) {
    workerWakeEvents_[i] = epicsEventCreate(epicsEventEmpty);
    workerDoneEvents_[i] = epicsEventCreate(epicsEventEmpty);
    sprintf(workerTaskName, "workerTask%d", i);
    pWCS = (workerCreateStruct *)malloc(sizeof(workerCreateStruct));
    pWCS->pTomoRecon = this;
    pWCS->taskNum = i;
    workerTaskId = epicsThreadCreate(workerTaskName,
                       epicsThreadPriorityMedium,
                       epicsThreadGetStackSize(epicsThreadStackMedium),
                       (EPICSTHREADFUNC) ::workerTask,
                       pWCS);
    if (workerTaskId == 0) {
      logMsg("%s epicsThreadCreate failure for workerTask %d", functionName, i);
      return;
    }
  } 
}

/** Destructor for the tomoRecon class.
* Calls shutDown() to stop any active reconstruction, which causes workerTasks to exit.
* Waits for supervisor task to exit, which in turn waits for all workerTasks to exit.
* Destroys the EPICS message queues, events and mutexes. Closes the debugging file. */
tomoRecon::~tomoRecon() 
{
  int i;
  int status;
  static const char *functionName = "tomoRecon:~tomoRecon";
  
  if (debug_) logMsg("%s: entry, shutting down and cleaning up", functionName);
  shutDown();
  status = epicsEventWait(supervisorDoneEvent_);
  if (status) {
    logMsg("%s: error waiting for supervisorDoneEvent=%d", functionName, status);
  }
  epicsMessageQueueDestroy(toDoQueue_);
  epicsMessageQueueDestroy(doneQueue_);
  epicsEventDestroy(supervisorWakeEvent_);
  epicsEventDestroy(supervisorDoneEvent_);
  for (i=0; i<numThreads_; i++) {
    epicsEventDestroy(workerWakeEvents_[i]);
    epicsEventDestroy(workerDoneEvents_[i]);
  }
  free(workerWakeEvents_);
  free(workerDoneEvents_);
  epicsMutexDestroy(fftwMutex_);
  if (debugFile_ != stdout) fclose(debugFile_);
}

/** Function to start reconstruction of a set of slices
* Sends messages to workerTasks to begin the reconstruction and wakes up
* the supervisorTasks and workerTasks.
* \param[in] numSlices Number of slices to reconstruct
* \param[in] center Rotation center to use for each slice
* \param[in] pInput Pointer to input data [numPixels, numSlices, numProjections]
* \param[out] pOutput Pointer to output data [numPixels, numPixels, numSlices] */
int tomoRecon::reconstruct(int numSlices, float *center, float *pInput, float *pOutput)
{
  float *pIn, *pOut;
  toDoMessage_t toDoMessage;
  int reconSize = numPixels_ * numPixels_;
  int nextSlice=0;
  int i;
  int status;
  static const char *functionName="tomoRecon::reconstruct";

  // If a reconstruction is already in progress return an error
  if (debug_) logMsg("%s: entry, reconComplete_=%d", functionName, reconComplete_);
  if (reconComplete_ == 0) {
    logMsg("%s: error, reconstruction already in progress", functionName);
    return -1;
  }

  if (numSlices > pTomoParams_->numSlices) {
    logMsg("%s: error, numSlices=%d, must be <= %d", functionName, numSlices, pTomoParams_->numSlices);
    return -1;
  }
  
  numSlices_ = numSlices;
  slicesRemaining_ = numSlices_;
  pInput_ = pInput;
  pOutput_ = pOutput;
  pIn = pInput_;
  pOut = pOutput_;

  reconComplete_ = 0;

  // Fill up the toDoQueue with slices to be reconstructed
  for (i=0; i<(numSlices_+1)/2; i++) {
    toDoMessage.sliceNumber = nextSlice;
    toDoMessage.pIn1 = pIn;
    toDoMessage.pOut1 = pOut;
    toDoMessage.center = center[i*2] + (paddedWidth_ - numPixels_)/2.;
    pIn += numPixels_;
    pOut += reconSize;
    nextSlice++;
    if (nextSlice < numSlices_) {
      toDoMessage.pIn2 = pIn;
      toDoMessage.pOut2 = pOut;
      pIn += numPixels_;
      pOut += reconSize;
      nextSlice++;
    } else {
      toDoMessage.pIn2 = NULL;
      toDoMessage.pOut2 = NULL;
    }
    status = epicsMessageQueueTrySend(toDoQueue_, &toDoMessage, sizeof(toDoMessage));
    if (status) {
      logMsg("%s:, error calling epicsMessageQueueTrySend, status=%d", 
          functionName, status);
    }
  }
  // Send events to start reconstruction
  if (debug_) logMsg("%s: sending events to start reconstruction", functionName);
  epicsEventSignal(supervisorWakeEvent_);
  for (i=0; i<numThreads_; i++) epicsEventSignal(workerWakeEvents_[i]);
  
  return 0;
}

/** Function to poll the status of the reconstruction
* \param[out] pReconComplete 0 if reconstruction is still in progress, 1 if it is complete
* \param[out] pSlicesRemaining Number of slices remaining to be reconstructed */
void tomoRecon::poll(int *pReconComplete, int *pSlicesRemaining)
{
  *pReconComplete = reconComplete_;
  *pSlicesRemaining = slicesRemaining_;
}

/** Function to shut down the object
* Sets the shutDown_ flag and sends an event to wake up the supervisorTask and workerTasks. */
void tomoRecon::shutDown()
{
  int i;
  
  // Set the shutdown flag
  shutDown_ = 1;
  // Send events to all threads waking them up
  epicsEventSignal(supervisorWakeEvent_);
  for (i=0; i<numThreads_; i++) epicsEventSignal(workerWakeEvents_[i]);
}

/** Supervisor control task that runs as a separate thread. 
* Reads messages from the workerTasks to update the status of the reconstruction (reconComplete and slicesRemaining).
* When shutting down waits for events from workerTasks threads indicating that they have all exited. */
void tomoRecon::supervisorTask()
{
  int i;
  int status;
  doneMessage_t doneMessage;
  static const char *functionName="tomoRecon::supervisorTask";

  while (1) {
    if (debug_) logMsg("%s: waiting for wake event", functionName);
    // Wait for a wake event
    epicsEventWait(supervisorWakeEvent_);
    if (shutDown_) goto done;
    // Wait for all the messages to come back, indicating that reconstruction is complete
    while (slicesRemaining_ > 0) {
      if (shutDown_) goto done;
      status = epicsMessageQueueReceive(doneQueue_, &doneMessage, sizeof(doneMessage));
      if (status != sizeof(doneMessage)) {
        logMsg("%s, error reading worker thread message", functionName);
        continue;
      }
      slicesRemaining_ -= doneMessage.numSlices;
    }
    if (debug_) logMsg("%s: All slices complete!", functionName);
    reconComplete_ = 1;
    if (debug_) logMsg("%s: Reconstruction complete!", functionName);
  }
  done:
  // Wait for the worker threads to exit before setting the reconstruction complete flag.
  // They will exit because the toDoQueue is now empty
  for (i=0; i<numThreads_; i++) {
    if (debug_) logMsg("%s: Beginning wait for worker task %d to complete, eventId=%p", 
          functionName, i, workerDoneEvents_[i]);
    status = epicsEventWaitWithTimeout(workerDoneEvents_[i], 1.0);
    if (status != epicsEventWaitOK) {
      logMsg("%s: Error waiting for worker task %d to complete, eventId=%p, status=%d", 
          functionName, i, workerDoneEvents_[i], status);
    }
  }
  // Send a signal to the destructor that the supervisor task is done
  if (debug_) logMsg("%s: Exiting supervisor task.", functionName);
  epicsEventSignal(supervisorDoneEvent_);
}

/** Worker task that runs as a separate thread. Multiple worker tasks can be running simultaneously.
 * Each workerTask thread reconstructs slices that it gets from the toDoQueue, and sends messages to
 * the supervisorTask via the doneQueue after reconstructing each pair of slices.
 * \param[in] taskNum Task number (0 to numThreads-1) for this tread; used to into arrays of event numbers in the object.
 */
void tomoRecon::workerTask(int taskNum)
{
  toDoMessage_t toDoMessage;
  doneMessage_t doneMessage;
  epicsEventId wakeEvent = workerWakeEvents_[taskNum];
  epicsEventId doneEvent = workerDoneEvents_[taskNum];
  epicsTimeStamp tStart, tStop;
  long reconSize;
  int imageSize;
  int status;
  float *pOut;
  int i;
  int sinOffset;
  float *sin1=0, *sin2=0, *recon1=0, *recon2=0, *pRecon;
  sg_struct sgStruct;
  grid_struct gridStruct;
  float **S1=0, **S2=0, **R1=0, **R2=0;
  float reconScale = pTomoParams_->reconScale;
  grid *pGrid=0;
  static const char *functionName="tomoRecon::workerTask";
  
  sgStruct.n_ang    = numProjections_;
  sgStruct.n_det    = paddedWidth_;
  // Force n_det to be odd
  if (paddedWidth_/2 != 0) sgStruct.n_det--;
  sgStruct.geom     = pTomoParams_->geom;
  sgStruct.angles   = pAngles_;
  sgStruct.center   = 0; // This is done per-slice
  get_pswf(pTomoParams_->pswfParam, &gridStruct.pswf);
  gridStruct.sampl     = pTomoParams_->sampl;
  gridStruct.R         = pTomoParams_->ROI;
  gridStruct.MaxPixSiz = pTomoParams_->MaxPixSiz;
  gridStruct.X0        = pTomoParams_->X0;
  gridStruct.Y0        = pTomoParams_->Y0;
  gridStruct.ltbl      = pTomoParams_->ltbl;
  gridStruct.filter    = get_filter(pTomoParams_->fname);
  gridStruct.verbose   = (debug_ > 1) ? 1 : 0;

  // Must take a mutex when creating grid object, because it creates fftw plans, which is not thread safe
  epicsMutexLock(fftwMutex_);
  if (debug_) logMsg("%s: %s creating grid object, filter=%s", 
                     functionName, epicsThreadGetNameSelf(), pTomoParams_->fname);
  pGrid = new grid(&gridStruct, &sgStruct, &reconSize);
  epicsMutexUnlock(fftwMutex_);

  sinOffset = (reconSize - numPixels_)/2;
  if (sinOffset < 0) sinOffset = 0;
  imageSize = reconSize;
  if (imageSize > numPixels_) imageSize = numPixels_;

  sin1   = (float *) calloc(paddedWidth_ * numProjections_, sizeof(float));
  sin2   = (float *) calloc(paddedWidth_ * numProjections_, sizeof(float));
  recon1 = (float *) calloc(reconSize * reconSize, sizeof(float));
  recon2 = (float *) calloc(reconSize * reconSize, sizeof(float));  
  S1     = (float **) malloc(numProjections_ * sizeof(float *));
  S2     = (float **) malloc(numProjections_ * sizeof(float *));
  R1     = (float **) malloc(reconSize * sizeof(float *));
  R2     = (float **) malloc(reconSize * sizeof(float *));

  /* We are passed addresses of arrays (float *), while Gridrec
     wants a pointer to a table of the starting address of each row.
     Need to build those tables */
  S1[0] = sin1;
  S2[0] = sin2;
  for (i=1; i<numProjections_; i++) {
    S1[i] = S1[i-1] + paddedWidth_;
    S2[i] = S2[i-1] + paddedWidth_;
  }
  R1[0] = recon1;
  R2[0] = recon2;
  for (i=1; i<reconSize; i++) {
      R1[i] = R1[i-1] + reconSize;
      R2[i] = R2[i-1] + reconSize;
  }

  while (1) {
    if (debug_) logMsg("%s: %s waiting for wake event", functionName, epicsThreadGetNameSelf());
    // Wait for an event signalling that reconstruction has started or exiting
    epicsEventWait(wakeEvent);
    if (shutDown_) goto done;
    while (1) {
      status = epicsMessageQueueTryReceive(toDoQueue_, &toDoMessage, sizeof(toDoMessage));
      if (status == -1) break;
      if (status != sizeof(toDoMessage)) {
        logMsg("%s:, error calling epicsMessageQueueReceive, status=%d", functionName, status);
        break;
      }
      epicsTimeGetCurrent(&tStart);

      sinogram(toDoMessage.pIn1, sin1);
      doneMessage.numSlices = 1;
      if (toDoMessage.pIn2) {
        sinogram(toDoMessage.pIn2, sin2);
        doneMessage.numSlices = 2;
      }
      epicsTimeGetCurrent(&tStop);
      doneMessage.sinogramTime = epicsTimeDiffInSeconds(&tStop, &tStart);
      epicsTimeGetCurrent(&tStart);
      pGrid->recon(toDoMessage.center, S1, S2, &R1, &R2);
      // Copy to output array, discard padding
      for (i=0, pOut=toDoMessage.pOut1, pRecon=recon1+sinOffset*reconSize; 
           i<imageSize;
           i++, pOut+=numPixels_, pRecon+=reconSize) {
        memcpy(pOut, pRecon+sinOffset, imageSize*sizeof(float));
      }
      // Multiply by reconScale
      if ((reconScale !=  0.) && (reconScale != 1.0)) {
        for (i=0, pOut=toDoMessage.pOut1; i<imageSize*imageSize; i++) {
          pOut[i] *= reconScale;
        }
      }
      if (doneMessage.numSlices == 2) {
        for (i=0, pOut=toDoMessage.pOut2, pRecon=recon2+sinOffset*reconSize; 
             i<imageSize;
             i++, pOut+=numPixels_, pRecon+=reconSize) {
          memcpy(pOut, pRecon+sinOffset, imageSize*sizeof(float));
        }
        // Multiply by reconScale
        if ((reconScale !=  0.) && (reconScale != 1.0)) {
          for (i=0, pOut=toDoMessage.pOut2; i<imageSize*imageSize; i++) {
            pOut[i] *= reconScale;
          }
        }
      }      
      epicsTimeGetCurrent(&tStop);
      doneMessage.reconTime = epicsTimeDiffInSeconds(&tStop, &tStart);
      doneMessage.sliceNumber = toDoMessage.sliceNumber;
      status = epicsMessageQueueTrySend(doneQueue_, &doneMessage, sizeof(doneMessage));
      if (status) {
        printf("%s, error calling epicsMessageQueueTrySend, status=%d", functionName, status);
      }
      if (debug_ > 0) { 
        logMsg("%s:, thread=%s, slice=%d, center=%f, sinogram time=%f, recon time=%f", 
            functionName, epicsThreadGetNameSelf(), doneMessage.sliceNumber, toDoMessage.center,
            doneMessage.sinogramTime, doneMessage.reconTime);
      }
      if (shutDown_) break;
    }
  }
  done:
  if (sin1) free(sin1);
  if (sin2) free(sin2);
  if (recon1) free(recon1);
  if (recon2) free(recon2);
  if (S1) free(S1);
  if (S2) free(S2);
  if (R1) free(R1);
  if (R2) free(R2);
  if (pGrid) delete pGrid;
  // Send an event so the supervisor knows this thread is done
  epicsEventSignal(doneEvent);
  if (debug_ > 0) {
    logMsg("tomoRecon::workerTask %s exiting, eventId=%p", 
        epicsThreadGetNameSelf(), doneEvent);
  }
}

/** Function to calculate a sinogram.
 * Takes log of data (unless fluorescence flag is set.
 * Optionally does secondary normalization to air in each row of sinogram.
 * Optionally does ring artifact reduction.
 * \param[in] pIn Pointer to normalized data input for this slice [numPixels, slice, numProjections]
 * \param[out] pOut Pointer to sinogram output [paddedSingramWidth, numProjections]
 */
void tomoRecon::sinogram(float *pIn, float *pOut)
{
  int i, j, k;
  int numAir = pTomoParams_->airPixels;
  int ringWidth = pTomoParams_->ringWidth;
  int sinOffset = (paddedWidth_ - numPixels_)/2;
  float *air=0, *averageRow=0, *smoothedRow=0;
  float airLeft, airRight, airSlope, ratio, outData;
  float *pInData;
  float *pOutData;
  //static const char *functionName = "tomoRecon::sinogram";
  
  if (numAir > 0) air = (float *) malloc(paddedWidth_*sizeof(float));
  if (ringWidth > 0) {
     averageRow = (float *) calloc(numPixels_, sizeof(float));
     smoothedRow = (float *) calloc(numPixels_, sizeof(float));
  }
  
  for (i=0, pInData=pIn, pOutData=pOut; 
       i<numProjections_;
       i++, pInData+=numPixels_*numSlices_, pOutData+=paddedWidth_) {
    if (numAir > 0) {
      for (j=0, airLeft=0, airRight=0; j<numAir; j++) {
        airLeft += pInData[j];
        airRight += pInData[numPixels_ - 1 - j];
      }
      airLeft /= numAir;
      airRight /= numAir;
      if (airLeft <= 0.) airLeft = 1.;
      if (airRight <= 0.) airRight = 1.;
      airSlope = (airRight - airLeft)/(numPixels_ - 1);
      for (j=0; j<numPixels_; j++) {
         air[j] = airLeft + airSlope*j;
      }
    }
    if (pTomoParams_->fluorescence) {
      for (j=0; j<numPixels_; j++) {
        pOutData[sinOffset + j] = pInData[j];
        if (ringWidth > 0) averageRow[j] += pInData[j];
      }
    }
    else {
      for (j=0; j<numPixels_; j++) {
        if (numAir > 0)
            ratio = pInData[j]/air[j];
        else
            ratio = pInData[j] * pTomoParams_->sinoScale;
        if (ratio <= 0.) ratio = 1.;
        outData = -log(ratio);
        pOutData[sinOffset + j] = outData;
        if (ringWidth > 0) averageRow[j] += outData;
      }
    }
  }
  // Do ring artifact correction if ringWidth > 0
  if (ringWidth > 0) {
    // We have now computed the average row of the sinogram
    // Smooth it
    for (i=0; i<numPixels_; i++) {
      averageRow[i] /= numProjections_;
    }
    for (i=0; i<numPixels_; i++) {
      smoothedRow[i] = 0;
      for (j=0; j<ringWidth; j++) {
        k = i+j-ringWidth/2;
        if (k < 0) k = 0;
        if (k > numPixels_ - 1) k = numPixels_ -1;
        smoothedRow[i] += averageRow[k];
      }
      smoothedRow[i] /= ringWidth;
    }
    // Subtract this difference from each row in sinogram
    for (i=0, pOutData=pOut; 
      i<numProjections_;
      i++, pOutData+=paddedWidth_) {
      for (j=0; j<numPixels_; j++) {
        pOutData[sinOffset + j] -= (averageRow[j] - smoothedRow[j]);
      }
    }
  }
  if (air) free(air);
  if (averageRow) free(averageRow);
  if (smoothedRow) free(smoothedRow);
}

/** Logs messages.
 * Adds time stamps to each message.
 * Does buffering to prevent messages from multiple threads getting garbled.
 * Adds the appropriate terminator for files (LF) and stdout (CR LF, needed for IDL on Linux).
 * Flushes output after each call, so output appears even if application crashes.
 * \param[in] pFormat Format string
 * \param[in] ... Additional arguments for vsprintf
 */
void tomoRecon::logMsg(const char *pFormat, ...)
{
  va_list     pvar;
  epicsTimeStamp now;
  char nowText[40];
  char message[256];
  char temp[256];

  epicsTimeGetCurrent(&now);
  nowText[0] = 0;
  epicsTimeToStrftime(nowText,sizeof(nowText),
      "%Y/%m/%d %H:%M:%S.%03f",&now);
  sprintf(message,"%s ",nowText);
  va_start(pvar,pFormat);
  vsprintf(temp, pFormat, pvar);
  va_end(pvar);
  strcat(message, temp);
  if (debugFile_ == stdout)
    strcat(message, "\r\n");
  else
    strcat(message, "\n");
  fprintf(debugFile_, message);
  fflush(debugFile_);
}

