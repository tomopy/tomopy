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
    int sizez;
    float pixel_size;
    } SimVars;

class DataSim {
public:
    DataSim(SimVars* pSimVars, float *pIn);
    void calc(int num_pts, float *srcx, float *srcy, float *srcz,
              float *detx, float *dety, float *detz, float *pOut);

private:
    float *pIn_;
    float *pOut_;
    SimVars *pSimVars_;
    int sizex_;
    int sizey_;
    int sizez_;
    float pixel_size_;
    int num_pts_;
    float xf_, xl_, yf_, yl_, zf_, zl_;
    float tmp;
    std::vector<float> xi_, yi_, zi_;
    std::vector<float> ax_, ay_, az_, axy_;
    std::vector<float> alpha_;
    std::vector<float> xk_, yk_, zk_;
    std::vector<float> dist_;
    int indx_, indy_, indz_;
    std::vector<int> ind_out_;
    float amin_, amax_;
    };
