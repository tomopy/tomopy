#include "dataSim.h"
using namespace std;

extern "C" {
    DataSim* create(SimVars* pSimVars, float *pIn) {
        return new DataSim(pSimVars, pIn);
        }

    void calc(DataSim *DataSim, int numPts, float *srcx,
              float *srcy, float *srcz, float *detx,
              float *dety, float *detz, float *pOut) {
        DataSim->calc(numPts, srcx, srcy, srcz,
                      detx, dety, detz, pOut);
        }
    } // extern "C"

DataSim::DataSim(SimVars* pSimVars, float *pIn) :
    pIn_(pIn),
    pSimVars_(pSimVars),
    sizex_(pSimVars->sizex),
    sizey_(pSimVars->sizey),
    sizez_(pSimVars->sizez),
    pixel_size_(pSimVars->pixel_size) {
    int m;
    for (m = 0; m < sizex_ + 1; m++) {
        xi_.push_back(pixel_size_ * (-float(sizex_) / 2 + m));
        }
    for (m = 0; m < sizey_ + 1; m++) {
        yi_.push_back(pixel_size_ * (-float(sizey_) / 2 + m));
        }
    for (m = 0; m < sizez_ + 1; m++) {
        zi_.push_back(pixel_size_ * (-float(sizez_) / 2 + m));
        }
    }

void DataSim::calc(int num_pts_, float *srcx,
                   float *srcy, float *srcz, float *detx,
                   float *dety, float *detz, float *pOut) {
    int m, n;
    for (m = 0; m < num_pts_; m++) {
        if (!ax_.empty()) { ax_.clear(); }
        if (!ay_.empty()) { ay_.clear(); }
        if (!az_.empty()) { az_.clear(); }
        if (!axy_.empty()) { axy_.clear(); }
        if (!alpha_.empty()) { alpha_.clear(); }
        if (!xk_.empty()) { xk_.clear(); }
        if (!yk_.empty()) { yk_.clear(); }
        if (!zk_.empty()) { zk_.clear(); }
        if (!dist_.empty()) { dist_.clear(); }
        if (!ind_out_.empty()) { ind_out_.clear(); }

        xf_ = (xi_[0] - srcx[m]) / (detx[m] - srcx[m]);
        xl_ = (xi_[sizex_] - srcx[m]) / (detx[m] - srcx[m]);
        yf_ = (yi_[0] - srcy[m]) / (dety[m] - srcy[m]);
        yl_ = (yi_[sizey_] - srcy[m]) / (dety[m] - srcy[m]);
        zf_ = (zi_[0] - srcz[m]) / (detz[m] - srcz[m]);
        zl_ = (zi_[sizez_] - srcz[m]) / (detz[m] - srcz[m]);

        amin_ = fmaxf(fmaxf(0, fminf(xf_, xl_)), fmaxf(fminf(yf_, yl_), fminf(zf_, zl_)));
        amax_ = fminf(fminf(1, fmaxf(xf_, xl_)), fminf(fmaxf(yf_, yl_), fmaxf(zf_, zl_)));

        for (n = 0; n < sizex_ + 1; n++) {
            tmp = (xi_[n] - srcx[m]) / (detx[m] - srcx[m]);
            if ((tmp >= amin_) && (tmp <= amax_)) {
                ax_.push_back(tmp);
                }
            }
        for (n = 0; n < sizey_ + 1; n++) {
            tmp = (yi_[n] - srcy[m]) / (dety[m] - srcy[m]);
            if ((tmp >= amin_) && (tmp <= amax_)) {
                ay_.push_back(tmp);
                }
            }
        for (n = 0; n < sizez_ + 1; n++) {
            tmp = (zi_[n] - srcz[m]) / (detz[m] - srcz[m]);
            if ((tmp >= amin_) && (tmp <= amax_)) {
                az_.push_back(tmp);
                }
            }
        std::merge(ax_.begin(), ax_.end(),
                   ay_.begin(), ay_.end(),
                   std::back_inserter(axy_));
        std::merge(axy_.begin(), axy_.end(),
                   az_.begin(), az_.end(),
                   std::back_inserter(alpha_));
        std::sort(alpha_.begin(), alpha_.end());
        alpha_.erase(unique(alpha_.begin(), alpha_.end()), alpha_.end());

        if (alpha_.size() > 1) {
            for (n = 0; n < alpha_.size(); n++) {
                xk_.push_back(srcx[m] + alpha_[n] * (detx[m] - srcx[m]));
                yk_.push_back(srcy[m] + alpha_[n] * (dety[m] - srcy[m]));
                zk_.push_back(srcz[m] + alpha_[n] * (detz[m] - srcz[m]));
                }

            for (n = 0; n < alpha_.size() - 1; n++) {
                dist_.push_back(sqrt(pow(xk_[n + 1] - xk_[n], 2) + pow(yk_[n + 1] - yk_[n], 2) + pow(zk_[n + 1] - zk_[n], 2)));
                }

            for (n = 0; n < dist_.size() - 1; n++) {
                indx_ = floor(((xk_[n] + xk_[n + 1]) / 2) / pixel_size_ + float(sizex_) / 2);
                indy_ = floor(((yk_[n] + yk_[n + 1]) / 2) / pixel_size_ + float(sizey_) / 2);
                indz_ = floor(((zk_[n] + zk_[n + 1]) / 2) / pixel_size_ + float(sizez_) / 2);
                ind_out_.push_back(indz_ + (indy_ + (indx_ * sizex_)) * sizey_);
                }

            for (n = 0; n < ind_out_.size(); n++) {
                pOut[m] += pIn_[ind_out_[n]] * dist_[n];
                }
            }
        }
    }
