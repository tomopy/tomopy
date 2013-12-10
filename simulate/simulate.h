#include <iostream>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <algorithm>
#include <iterator>
#include <vector>
#include <iterator>
#include <cstdlib>

const double PI = 3.141592653589793238462;

typedef struct {
    int sizex;
    int sizey;
    float pixel_size;
    float energy;
    } Source;

typedef struct {
    int sizex;
    int sizey;
    float pixel_size;
    } Detector;

typedef struct {
    int sizex;
    int sizey;
    int sizez;
    float pixel_size;
    } Phantom;

class Simulate {
public:
    Simulate(Source *pSource,
             Detector *pDetector,
             Phantom* pPhantom,
             float *pinput);
    void calc3d(float *psrcx, float *psrcy, float *psrcz,
                float *pdetx, float *pdety, float *pdetz,
                float *poutput);
    void calc2d(float *psrcx, float *psrcy,
                float *pdetx, float *pdety,
                float *poutput);

private:
    float *pinput_;

    Source *pSource_;
    int src_sizex_;
    int src_sizey_;
    float src_pixel_size_;
    float src_energy;

    Detector *pDetector_;
    int det_sizex_;
    int det_sizey_;
    float det_pixel_size_;

    Phantom *pPhantom_;
    int obj_sizex_;
    int obj_sizey_;
    int obj_sizez_;
    float obj_pixel_size_;

    int num_pts_;
    float xf_, xl_, yf_, yl_, zf_, zl_;
    float tmp;
    std::vector<float> xi_, yi_, zi_;
    std::vector<float> ax_, ay_, az_, axy_;
    std::vector<float> alpha_;
    std::vector<float> xk_, yk_, zk_;
    std::vector<float> dist_;
    int indx_, indy_, indz_;
    int ind_out_;
    float amin_, amax_;
    };
