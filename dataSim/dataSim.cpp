#include "dataSimTest.h"
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

dataSim::dataSim(simVars* pSimVars, float *pIn) :
    vpIn(pIn),
    vpSimVars(pSimVars),
    vObjSizeX(pSimVars -> objSizeX),
    vObjSizeY(pSimVars -> objSizeY),
    vObjSizeZ(pSimVars -> objSizeZ),
    vObjPixelSize(pSimVars -> objPixelSize) {
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
    for (m = 0; m < numPts; m++) {
        if (!vAx.empty()) { vAx.clear(); }
        if (!vAy.empty()) { vAy.clear(); }
        if (!vAz.empty()) { vAz.clear(); }
        if (!vAxy.empty()) { vAxy.clear(); }
        if (!vAlpha.empty()) { vAlpha.clear(); }
        if (!vXk.empty()) { vXk.clear(); }
        if (!vYk.empty()) { vYk.clear(); }
        if (!vZk.empty()) { vZk.clear(); }
        if (!vDist.empty()) { vDist.clear(); }
        if (!vIndOut.empty()) { vIndOut.clear(); }

        vxf = (vXi[0] - srcx[m]) / (detx[m] - srcx[m]);
        vxl = (vXi[vObjSizeX] - srcx[m]) / (detx[m] - srcx[m]);
        vyf = (vYi[0] - srcy[m]) / (dety[m] - srcy[m]);
        vyl = (vYi[vObjSizeY] - srcy[m]) / (dety[m] - srcy[m]);
        vzf = (vZi[0] - srcz[m]) / (detz[m] - srcz[m]);
        vzl = (vZi[vObjSizeZ] - srcz[m]) / (detz[m] - srcz[m]);

        vAmin = fmaxf(fmaxf(0, fminf(vxf, vxl)), fmaxf(fminf(vyf, vyl), fminf(vzf, vzl)));
        vAmax = fminf(fminf(1, fmaxf(vxf, vxl)), fminf(fmaxf(vyf, vyl), fmaxf(vzf, vzl)));

        for (n = 0; n < vObjSizeX + 1; n++) {
            tmp = (vXi[n] - srcx[m]) / (detx[m] - srcx[m]);
            if ((tmp >= vAmin) && (tmp <= vAmax)) {
                vAx.push_back(tmp);
                }
            }
        for (n = 0; n < vObjSizeY + 1; n++) {
            tmp = (vYi[n] - srcy[m]) / (dety[m] - srcy[m]);
            if ((tmp >= vAmin) && (tmp <= vAmax)) {
                vAy.push_back(tmp);
                }
            }
        for (n = 0; n < vObjSizeZ + 1; n++) {
            tmp = (vZi[n] - srcz[m]) / (detz[m] - srcz[m]);
            if ((tmp >= vAmin) && (tmp <= vAmax)) {
                vAz.push_back(tmp);
                }
            }
        std::merge(vAx.begin(), vAx.end(),
                   vAy.begin(), vAy.end(),
                   std::back_inserter(vAxy));
        std::merge(vAxy.begin(), vAxy.end(),
                   vAz.begin(), vAz.end(),
                   std::back_inserter(vAlpha));
        std::sort(vAlpha.begin(), vAlpha.end());
        vAlpha.erase(unique(vAlpha.begin(), vAlpha.end()), vAlpha.end());

        if (vAlpha.size() > 1) {
            for (n = 0; n < vAlpha.size(); n++) {
                vXk.push_back(srcx[m] + vAlpha[n] * (detx[m] - srcx[m]));
                vYk.push_back(srcy[m] + vAlpha[n] * (dety[m] - srcy[m]));
                vZk.push_back(srcz[m] + vAlpha[n] * (detz[m] - srcz[m]));
                }

            for (n = 0; n < vAlpha.size() - 1; n++) {
                vDist.push_back(sqrt(pow(vXk[n + 1] - vXk[n], 2) + pow(vYk[n + 1] - vYk[n], 2) + pow(vZk[n + 1] - vZk[n], 2)));
                }

            for (n = 0; n < vDist.size() - 1; n++) {
                vIndx = floor(((vXk[n] + vXk[n + 1]) / 2) / vObjPixelSize + float(vObjSizeX) / 2);
                vIndy = floor(((vYk[n] + vYk[n + 1]) / 2) / vObjPixelSize + float(vObjSizeY) / 2);
                vIndz = floor(((vZk[n] + vZk[n + 1]) / 2) / vObjPixelSize + float(vObjSizeZ) / 2);
                vIndOut.push_back(vIndz + (vIndy + (vIndx * vObjSizeX)) * vObjSizeY);
                }

            for (n = 0; n < vIndOut.size(); n++) {
                pOut[m] += vpIn[vIndOut[n]] * vDist[n];
                }
            }
        }
    }
