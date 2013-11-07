#include "tomoRecon.h"
#include <epicsExport.h>


static tomoRecon *pTomoRecon = NULL;
static tomoParams_t tomoParams;
static float *angles = NULL;

extern "C" {
epicsShareFunc void epicsShareAPI reconCreate(tomoParams_t *pTomoParams, float *pAngles)
{
  if (angles) free(angles);
  angles = NULL;
  if (pTomoRecon) delete pTomoRecon;
  pTomoRecon = NULL;
  memcpy(&tomoParams, pTomoParams, sizeof(tomoParams));
  angles = (float *)malloc(tomoParams.numProjections*sizeof(float));
  memcpy(angles, pAngles, tomoParams.numProjections*sizeof(float));
  pTomoRecon = new tomoRecon(&tomoParams, angles);
}

epicsShareFunc void epicsShareAPI reconDelete()
{
  if (pTomoRecon) delete pTomoRecon;
  pTomoRecon = NULL;
  if (angles) free(angles);
  angles = NULL;
}

epicsShareFunc void epicsShareAPI reconRun(int *numSlices, 
                                          float *pCenter, 
                                          float *pIn, 
                                          float *pOut)
{
  if (pTomoRecon == NULL) return;
  pTomoRecon -> reconstruct(*numSlices, pCenter, pIn, pOut);
}

epicsShareFunc void epicsShareAPI reconPoll(int *pReconComplete,
                                            int *pSlicesRemaining)
{
  if (pTomoRecon == NULL) return;
  pTomoRecon -> poll(pReconComplete, pSlicesRemaining);
}

} // extern "C"
