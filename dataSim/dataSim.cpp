#include "dataSim.h"
using namespace std;

extern "C" {
    dataSim* create(simVars* pSimVars, float *pIn) {
        return new dataSim(pSimVars, pIn);
        }

    void calc(dataSim *dataSim, int numPts, float *srcx,
              float *srcy, float *srcz, float *detx,
              float *dety, float *detz, float *pOut) {
        dataSim -> calc(numPts, srcx, srcy, srcz,
                        detx, dety, detz, pOut);
        }
    } // extern "C"

dataSim::dataSim(simVars* pSimVars, float *pIn) {
    std::cout << "Grr!!" << std::endl;
    vpIn = pIn;
    vpSimVars = pSimVars;
    vObjSizeX = pSimVars -> objSizeX;
    vObjSizeY = pSimVars -> objSizeY;
    vObjSizeZ = pSimVars -> objSizeZ;
    vObjPixelSize = pSimVars -> objPixelSize;

    for (m = 0; m < vObjSizeX + 1; m++) {
        vXi.push_back(vObjPixelSize * (-float(vObjSizeX) / 2 + m));
        }
    for (m = 0; m < vObjSizeY + 1; m++) {
        vYi.push_back(vObjPixelSize * (-float(vObjSizeY) / 2 + m));
        }
    for (m = 0; m < vObjSizeZ + 1; m++) {
        vZi.push_back(vObjPixelSize * (-float(vObjSizeZ) / 2 + m));
        }
    }

void dataSim::calc(int numPts, float *srcx,
                   float *srcy, float *srcz, float *detx,
                   float *dety, float *detz, float *pOut) {
    std::cout << "Ugh!!" << std::endl;
    for (m = 0; m < numPts; m++) {
        if (!vAx.empty()) {
            vAx.clear();
            }
        if (!vAy.empty()) {
            vAy.clear();
            }
        if (!vAz.empty()) {
            vAz.clear();
            }
        if (!vAxIn.empty()) {
            vAxIn.clear();
            }
        if (!vAyIn.empty()) {
            vAyIn.clear();
            }
        if (!vAzIn.empty()) {
            vAzIn.clear();
            }
        if (!vAlpha.empty()) {
            vAlpha.clear();
            }
        if (!vXk.empty()) {
            vXk.clear();
            }
        if (!vYk.empty()) {
            vYk.clear();
            }
        if (!vZk.empty()) {
            vZk.clear();
            }
        if (!vDist.empty()) {
            vDist.clear();
            }
        if (!vIndOut.empty()) {
            vIndOut.clear();
            }

        // Calculate the alpha values.
        for (n = 0; n < vObjSizeX + 1; n++) {
            vAx.push_back((vXi[n] - srcx[m]) / (detx[m] - srcx[m]));
            }
        for (n = 0; n < vObjSizeY + 1; n++) {
            vAy.push_back((vYi[n] - srcy[m]) / (dety[m] - srcy[m]));
            }
        for (n = 0; n < vObjSizeZ + 1; n++) {
            vAz.push_back((vZi[n] - srcz[m]) / (detz[m] - srcz[m]));
            }

        vAmin = fmaxf(fmaxf(0, fminf(vAx[0], vAx[vObjSizeX])), fmaxf(fminf(vAy[0], vAy[vObjSizeY]), fminf(vAz[0], vAz[vObjSizeZ])));
        vAmax = fminf(fminf(1, fmaxf(vAx[0], vAx[vObjSizeX])), fminf(fmaxf(vAy[0], vAy[vObjSizeY]), fmaxf(vAz[0], vAz[vObjSizeZ])));

        //std::cout << vAmax << std::endl;
        for (n = 0; n < vObjSizeX + 1; n++) {
            if ((vAx[n] >= vAmin) && (vAx[n] <= vAmax)) {
                vAxIn.push_back(vAx[n]);
                }
            }
        for (n = 0; n < vObjSizeY + 1; n++) {
            if ((vAy[n] >= vAmin) && (vAy[n] <= vAmax)) {
                vAyIn.push_back(vAy[n]);
                }
            }
        for (n = 0; n < vObjSizeZ + 1; n++) {
            if ((vAz[n] >= vAmin) && (vAz[n] <= vAmax)) {
                vAzIn.push_back(vAz[n]);
                }
            }
        std::merge(vAxIn.begin(), vAxIn.end(),
                   vAyIn.begin(), vAyIn.end(),
                   std::back_inserter(vAlpha));
        std::sort(vAlpha.begin(), vAlpha.end());

        if (vAlpha.size() > 1) {
            for (n = 0; n < vAlpha.size(); n++) {
                vXk.push_back(srcx[m] + vAlpha[n] * (detx[m] - srcx[m]));
                vYk.push_back(srcy[m] + vAlpha[n] * (dety[m] - srcy[m]));
                vZk.push_back(srcz[m] + vAlpha[n] * (detz[m] - srcz[m]));
                }

            for (n = 0; n < vAlpha.size() - 1; n++) {
                vDist.push_back(sqrt(pow(vXk[n + 1] - vXk[n], 2) + pow(vYk[n + 1] - vYk[n], 2) +  + pow(vZk[n + 1] - vZk[n], 2)));
                }

            for (n = 0; n < vDist.size() - 1; n++) {
                vIndx = floor(((vXk[n] + vXk[n + 1]) / 2) / vObjPixelSize + float(vObjSizeX) / 2);
                vIndy = floor(((vYk[n] + vYk[n + 1]) / 2) / vObjPixelSize + float(vObjSizeY) / 2);
                vIndz = floor(((vZk[n] + vZk[n + 1]) / 2) / vObjPixelSize + float(vObjSizeZ) / 2);
                vIndOut.push_back(vIndz + (vIndy + (vIndx * vObjSizeX)) * vObjSizeY);
                }
            //for (n = 0; n < vDist.size(); n++) {
            //    std::cout << vDist[n] << std::endl;
            //    }

            for (n = 0; n < vIndOut.size(); n++) {
                pOut[m] += vpIn[vIndOut[n]] * vDist[n];
                }
            }
        //std::cout << pOut[m] << std::endl;
        }
    }
