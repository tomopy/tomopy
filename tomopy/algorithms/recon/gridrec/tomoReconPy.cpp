#include "tomoRecon.h"
#include <boost/thread/thread.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <functional>
#include <iostream>

static tomoRecon *pTomoRecon = NULL;
static tomoParams_t tomoParams;
static float *angles = NULL;

extern "C" {

void reconCreate(tomoParams_t *pTomoParams, float *pAngles)
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

void reconDelete()
{
    if (pTomoRecon) delete pTomoRecon;
    pTomoRecon = NULL;
    if (angles) free(angles);
    angles = NULL;
}

void reconRun(int *numSlices,
             float *pCenter,
             float *pIn,
             float *pOut)
{
    if (pTomoRecon == NULL) return;
    pTomoRecon -> reconstruct(*numSlices, pCenter, pIn, pOut);
}

void reconPoll(int *pReconComplete,
               int *pSlicesRemaining)
{
    if (pTomoRecon == NULL) return;
    pTomoRecon -> poll(pReconComplete, pSlicesRemaining);
}

} // extern "C"