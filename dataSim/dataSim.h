#include <iostream>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <algorithm>
#include <iterator>
#include <vector>
#include <iterator>
#include <cstdlib>

//In the code given below:
//t stands for protected,
//v stands for private,
//m stands for member-variable
//p stands for pointer.

const double PI = 3.141592653589793238462;

typedef struct {
    int objSizeX;
    int objSizeY;
    int objSizeZ;
    float objPixelSize;
    } simVars;

class dataSim {
public:
    dataSim(simVars* params, float *pIn);
    void calc(int numPts, float *srcx, float *srcy, float *srcz,
              float *detx, float *dety, float *detz, float *pOut);

private:
    int m, n;
    float *vpIn;
    float *vpOut;
    simVars *vpSimVars;
    int vObjSizeX;
    int vObjSizeY;
    int vObjSizeZ;
    float vObjPixelSize;
    int vNumPts;
    float vxf, vxl, vyf, vyl, vzf, vzl;
    float tmp;
    std::vector<float> vXi, vYi, vZi;
    std::vector<float> vAx, vAy, vAz, vAxy;
    std::vector<float> vAlpha;
    std::vector<float> vXk, vYk, vZk;
    std::vector<float> vDist;
    int vIndx, vIndy, vIndz;
    std::vector<int> vIndOut;
    float vAmin, vAmax;
    };
