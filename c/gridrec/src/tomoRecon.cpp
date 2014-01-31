/*-----------------------------------------------------------------------------
 * Copyright (c) 2013, UChicago Argonne, LLC
 * See LICENSE file.
 *---------------------------------------------------------------------------*/

#include <stdlib.h>
#include <math.h>
#include <stdarg.h>
#include <iostream>
#include "tomoRecon.h"
#include <boost/bind.hpp>
#include <boost/thread.hpp>

/*---------------------------------------------------------------------------*/

tomoRecon::tomoRecon(tomoParams_t *pTomoParams, float *pAngles)
:  pTomoParams_(pTomoParams),
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

   char workerTaskName[20];
   char *debugFileName = pTomoParams_->debugFileName;
   int i;
   static const char *functionName="tomoRecon::tomoRecon";

   debugFile_ = stdout;
   if ((debugFileName) && (strlen(debugFileName) > 0)) {
     debugFile_ = fopen(debugFileName, "w");
   }
  
   if (debug_) logMsg("%s: entry, creating message queues", functionName);

   toDoQueue_ = toDoMsgQueue.MessageQueueCreate(queueElements_,
                                                sizeof(toDoMessage_t));
   doneQueue_ = doneMsgQueue.MessageQueueCreate(queueElements_,
                                                sizeof(doneMessage_t));

}

/*---------------------------------------------------------------------------*/

tomoRecon::~tomoRecon() 
{
   int i;
   int status;
   static const char *functionName = "tomoRecon:~tomoRecon";

   if (debug_) logMsg("%s: entry, shutting down and cleaning up", functionName);

   toDoMsgQueue.MessageQueueDestroy(toDoQueue_);
   doneMsgQueue.MessageQueueDestroy(doneQueue_);

   if (debugFile_ != stdout) fclose(debugFile_);
}

/*---------------------------------------------------------------------------*/

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
   if (reconComplete_ == 0)
   {
     logMsg("%s: error, reconstruction already in progress", functionName);
     return -1;
   }

   if (numSlices > pTomoParams_->numSlices)
   {
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
   for (i=0; i<(numSlices_+1)/2; i++)
   {
     toDoMessage.sliceNumber = nextSlice;
     toDoMessage.pIn1 = pIn;
     toDoMessage.pOut1 = pOut;
     toDoMessage.center = center[i*2] + (paddedWidth_ - numPixels_)/2.;
     pIn += numPixels_;
     pOut += reconSize;
     nextSlice++;
     if (nextSlice < numSlices_)
     {
       toDoMessage.pIn2 = pIn;
       toDoMessage.pOut2 = pOut;
       pIn += numPixels_;
       pOut += reconSize;
       nextSlice++;
     } else
     {
       toDoMessage.pIn2 = NULL;
       toDoMessage.pOut2 = NULL;
     }

     status = toDoMsgQueue.MessageQueueTrySend(toDoQueue_, &toDoMessage, sizeof(toDoMessage));
     if (status)
     {
       logMsg("%s:, error calling MessageQueueTrySend, status=%d",
           functionName, status);
     }
   }

   // WOrker threads started here
   boost::thread** workerThreads;

   workerThreads = new boost::thread*[numThreads_];

   for(int i = 0; i < numThreads_; i++)
   {
      workerThreads[i] = new boost::thread(boost::bind(&tomoRecon::workerTask, this,i));
   }

   //Wait for all the worker threads to finish
   for(int i = 0; i < numThreads_; i++)
   {
      workerThreads[i]->join();
   }

   //Free worker threads memory
   for(int i = 0; i < numThreads_; i++)
   {
      delete workerThreads[i];
   }
   delete[] workerThreads;

   reconComplete_ = 1;
   slicesRemaining_ = 0;
  
   return 0;
}

/*---------------------------------------------------------------------------*/

void tomoRecon::poll(int *pReconComplete, int *pSlicesRemaining)
{
   *pReconComplete = reconComplete_;
   *pSlicesRemaining = slicesRemaining_;
}

/*---------------------------------------------------------------------------*/

void tomoRecon::workerTask(int taskNum)
{

   toDoMessage_t toDoMessage;
   doneMessage_t doneMessage;

   boost::posix_time::ptime tStart, tStop;
   boost::posix_time::time_duration diff;

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

   m_mutex.lock();

   pGrid = new grid(&gridStruct, &sgStruct, &reconSize);

   m_mutex.unlock();

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

   for (i=1; i<numProjections_; i++)
   {
     S1[i] = S1[i-1] + paddedWidth_;
     S2[i] = S2[i-1] + paddedWidth_;
   }
   R1[0] = recon1;
   R2[0] = recon2;

   for (i=1; i<reconSize; i++)
   {
       R1[i] = R1[i-1] + reconSize;
       R2[i] = R2[i-1] + reconSize;
   }

   while (slicesRemaining_ != 0)
   {
      status = toDoMsgQueue.MessageQueueTryReceive(toDoQueue_, &toDoMessage, sizeof(toDoMessage));
      if (status == -1) break;
      if (status != sizeof(toDoMessage))
      {
         logMsg("%s:, error calling epicsMessageQueueReceive, status=%d", functionName, status);
         break;
      }

      //epicsTimeGetCurrent(&tStart);
      tStart = boost::posix_time::second_clock::local_time();
      sinogram(toDoMessage.pIn1, sin1);
      doneMessage.numSlices = 1;

      if (toDoMessage.pIn2)
      {
         sinogram(toDoMessage.pIn2, sin2);
         doneMessage.numSlices = 2;
      }

      m_mutex.lock();
      slicesRemaining_ -= doneMessage.numSlices;
      m_mutex.unlock();

      tStop = boost::posix_time::second_clock::local_time();
      diff = tStop - tStart;
      doneMessage.sinogramTime = diff.total_milliseconds();

      tStart = boost::posix_time::second_clock::local_time();
      pGrid->recon(toDoMessage.center, S1, S2, &R1, &R2);
      // Copy to output array, discard padding
      for (i=0, pOut=toDoMessage.pOut1, pRecon=recon1+sinOffset*reconSize;
           i<imageSize;
           i++, pOut+=numPixels_, pRecon+=reconSize)
      {
         memcpy(pOut, pRecon+sinOffset, imageSize*sizeof(float));
      }
      // Multiply by reconScale
      if ((reconScale !=  0.) && (reconScale != 1.0))
      {
         for (i=0, pOut=toDoMessage.pOut1; i<imageSize*imageSize; i++)
         {
            pOut[i] *= reconScale;
         }
      }
      if (doneMessage.numSlices == 2)
      {
         for (i=0, pOut=toDoMessage.pOut2, pRecon=recon2+sinOffset*reconSize;
             i<imageSize;
             i++, pOut+=numPixels_, pRecon+=reconSize)
         {
            memcpy(pOut, pRecon+sinOffset, imageSize*sizeof(float));
         }

         // Multiply by reconScale
         if ((reconScale !=  0.) && (reconScale != 1.0))
         {
            for (i=0, pOut=toDoMessage.pOut2; i<imageSize*imageSize; i++)
            {
               pOut[i] *= reconScale;
            }
         }
      }

      tStop = boost::posix_time::second_clock::local_time();
      diff = tStop - tStart;
      doneMessage.reconTime = diff.total_milliseconds();

      doneMessage.sliceNumber = toDoMessage.sliceNumber;
      status = doneMsgQueue.MessageQueueTrySend(doneQueue_, &doneMessage, sizeof(doneMessage));

      if (status)
      {
         printf("%s, error calling epicsMessageQueueTrySend, status=%d", functionName, status);
      }

      if (0==slicesRemaining_) break;
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

}

/*---------------------------------------------------------------------------*/

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
   static const char *functionName = "tomoRecon::sinogram";

   if (numAir > 0) air = (float *) malloc(paddedWidth_*sizeof(float));
   if (ringWidth > 0)
   {
      averageRow = (float *) calloc(numPixels_, sizeof(float));
      smoothedRow = (float *) calloc(numPixels_, sizeof(float));
   }
  
   for (i=0, pInData=pIn, pOutData=pOut;
        i<numProjections_;
        i++, pInData+=numPixels_*numSlices_, pOutData+=paddedWidth_)
   {
      if (numAir > 0)
      {
         for (j=0, airLeft=0, airRight=0; j<numAir; j++)
         {
            airLeft += pInData[j];
            airRight += pInData[numPixels_ - 1 - j];
         }
         airLeft /= numAir;
         airRight /= numAir;
         if (airLeft <= 0.) airLeft = 1.;
         if (airRight <= 0.) airRight = 1.;
         airSlope = (airRight - airLeft)/(numPixels_ - 1);

         for (j=0; j<numPixels_; j++)
         {
            air[j] = airLeft + airSlope*j;
         }
      }

      if (pTomoParams_->fluorescence)
      {
         for (j=0; j<numPixels_; j++)
         {
            pOutData[sinOffset + j] = pInData[j];
            if (ringWidth > 0) averageRow[j] += pInData[j];
         }
      }
      else
      {
         for (j=0; j<numPixels_; j++)
         {
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
   if (ringWidth > 0)
   {
      // We have now computed the average row of the sinogram
      // Smooth it
      for (i=0; i<numPixels_; i++)
      {
         averageRow[i] /= numProjections_;
      }
      for (i=0; i<numPixels_; i++)
      {
         smoothedRow[i] = 0;
         for (j=0; j<ringWidth; j++)
         {
            k = i+j-ringWidth/2;
            if (k < 0) k = 0;
            if (k > numPixels_ - 1) k = numPixels_ -1;
            smoothedRow[i] += averageRow[k];
         }
            smoothedRow[i] /= ringWidth;
      }
      // Subtract this difference from each row in sinogram
      for (i=0, pOutData=pOut; i<numProjections_; i++, pOutData+=paddedWidth_)
      {
         for (j=0; j<numPixels_; j++)
         {
            pOutData[sinOffset + j] -= (averageRow[j] - smoothedRow[j]);
         }
      }
   }

   if (air) free(air);
   if (averageRow) free(averageRow);
   if (smoothedRow) free(smoothedRow);

}

/*---------------------------------------------------------------------------*/

void tomoRecon::logMsg(const char *pFormat, ...)
{

   va_list pvar;
   char nowText[40];
   char message[256];
   char temp[256];

   boost::posix_time::ptime now;
   now = boost::posix_time::second_clock::local_time();

   nowText[0] = 0;
   sprintf(message,"%s ",to_simple_string(now).c_str());
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

/*---------------------------------------------------------------------------*/

