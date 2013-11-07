#include <iostream>
#include <math.h>
#include <stdlib.h>
using namespace std;

extern "C" {
void nearestFineToCoarse(float *imgIn,
                        float *point, 
                        int *npoint,
                        int *resolution,
                        float *limits,
                        float *imgOut,
                        float *counter) {
    int m;    
    float coordx, coordy;
    float newCoordx, newCoordy;
    float dx, dy;
    float w0, w1, w2, w3;
    int ind;
    int indX, indY;
    int ngrid;
    
    float sx = (limits[1] - limits[0]) / resolution[0];
    float sy = (limits[3] - limits[2]) / resolution[1];
    float tx = limits[0] - (sx / 2);
    float ty = limits[2] - (sy / 2);
    
    for (m = 0; m < *npoint; m++) {
        coordx = point[2*m];
        coordy = point[2*m+1];
        
        newCoordx = (coordx - tx) / sx;
        newCoordy = (coordy - ty) / sy;
        
        indX = round(newCoordx);
        indY = round(newCoordy);
        
        ind = indY + (indX * (resolution[1]+2));
    
        imgOut[ind] += imgIn[m];
        counter[ind] += 1;
        }
    
    ngrid = (resolution[0]+2) * (resolution[1]+2);
    for (m = 0; m < ngrid; m++) {
        if (counter[m] != 0) {
            imgOut[m] = imgOut[m] / counter[m];
            }
        }
    }

void bilinearFineToCoarse(float *imgIn,
                          float *point, 
                          int *npoint,
                          int *resolution,
                          float *limits,
                          float *imgOut,
                          float *counter) {
    int m;
    float coordx, coordy;
    float newCoordx, newCoordy;
    float dx, dy;
    float w0, w1, w2, w3;
    int ind0, ind1, ind2, ind3;
    int indFloorX, indCeilX, indFloorY, indCeilY;
    int ngrid;
    
    float sx = (limits[1] - limits[0]) / resolution[0];
    float sy = (limits[3] - limits[2]) / resolution[1];
    float tx = limits[0] - (sx / 2);
    float ty = limits[2] - (sy / 2);
    
    for (m = 0; m < *npoint; m++) {
        coordx = point[2*m];
        coordy = point[2*m+1];
        
        newCoordx = (coordx - tx) / sx;
        newCoordy = (coordy - ty) / sy;
        
        indFloorX = floor(newCoordx);
        indCeilX = ceil(newCoordx);
        indFloorY = floor(newCoordy);
        indCeilY = ceil(newCoordy);
        
        dx = indCeilX - newCoordx;
        dy = indCeilY - newCoordy;
                    
        w0 = (1 - dx) * (1 - dy);
        w1 = (1 - dx) * dy;
        w2 = dx * (1 - dy);
        w3 = dx * dy;
    
        ind0 = indCeilY + (indCeilX * (resolution[1]+2));
        ind1 = indCeilY + (indFloorX * (resolution[1]+2));
        ind2 = indFloorY + (indCeilX * (resolution[1]+2));
        ind3 = indFloorY + (indFloorX * (resolution[1]+2));
    
        imgOut[ind0] += w0 * imgIn[m];
        imgOut[ind1] += w1 * imgIn[m];
        imgOut[ind2] += w2 * imgIn[m];
        imgOut[ind3] += w3 * imgIn[m];
    
        counter[ind0] += w0;
        counter[ind1] += w1;
        counter[ind2] += w2;
        counter[ind3] += w3;
    }
    
    ngrid = (resolution[0]+2) * (resolution[1]+2);
    for (m = 0; m < ngrid; m++) {
        if (counter[m] != 0) {
            imgOut[m] = imgOut[m] / counter[m];
            }
        }
    }

void nearestInterp2d(float *imgIn,
                    float *point, 
                    int *npoint,
                    int *resolution,
                    float *limits,
                    float *imgOut) {
    float coordx, coordy;
    float newCoordx, newCoordy;
    float dx, dy;
    float w0, w1, w2, w3;
    int ind;
    int indX, indY;
    
    float sx = (limits[1] - limits[0]) / resolution[0];
    float sy = (limits[3] - limits[2]) / resolution[1];
    float tx = limits[0] - (sx / 2);
    float ty = limits[2] - (sy / 2);
    
    for (int m = 0; m < *npoint; m++) {
        coordx = point[2*m];
        coordy = point[2*m+1];
        
        newCoordx = (coordx - tx) / sx;
        newCoordy = (coordy - ty) / sy;
        
        indX = round(newCoordx);
        indY = round(newCoordy);
        
        ind = indY + (indX * (resolution[1]+2));
    
        imgOut[m] += imgIn[ind];
        }
    }

void bilinearInterp2d(float *imgIn,
                     float *point, 
                     int *npoint,
                     int *resolution,
                     float *limits,
                     float *imgOut) {
    float coordx, coordy;
    float newCoordx, newCoordy;
    float dx, dy;
    float w0, w1, w2, w3;
    int ind0, ind1, ind2, ind3;
    int indFloorX, indCeilX, indFloorY, indCeilY;
    
    float sx = (limits[1] - limits[0]) / resolution[0];
    float sy = (limits[3] - limits[2]) / resolution[1];
    float tx = limits[0] - (sx / 2);
    float ty = limits[2] - (sy / 2);
    
    for (int m = 0; m < *npoint; m++) {
        coordx = point[2*m];
        coordy = point[2*m+1];
        
        newCoordx = (coordx - tx) / sx;
        newCoordy = (coordy - ty) / sy;
        
        indFloorX = floor(newCoordx);
        indCeilX = ceil(newCoordx);
        indFloorY = floor(newCoordy);
        indCeilY = ceil(newCoordy);
        
        dx = indCeilX - newCoordx;
        dy = indCeilY - newCoordy;
                    
        w0 = (1 - dx) * (1 - dy);
        w1 = (1 - dx) * dy;
        w2 = dx * (1 - dy);
        w3 = dx * dy;
        
        ind0 = indCeilY + (indCeilX * (resolution[1]+2));
        ind1 = indCeilY + (indFloorX * (resolution[1]+2));
        ind2 = indFloorY + (indCeilX * (resolution[1]+2));
        ind3 = indFloorY + (indFloorX * (resolution[1]+2));
        imgOut[m] = w0*imgIn[ind0] + w1*imgIn[ind1] + w2*imgIn[ind2] + w3*imgIn[ind3];
        }
    }
} // extern "C"

