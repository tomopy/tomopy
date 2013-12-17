#include "simulate.h"
using namespace std;

extern "C" {
    Simulate* create(Source* pSource, Detector* pDetector,
                     Phantom* pPhantom, float *pinput) {
        return new Simulate(pSource, pDetector, pPhantom, pinput);
        }

    void calc3d(Simulate *Simulate,
                float *psrcx, float *psrcy, float *psrcz,
                float *pdetx, float *pdety, float *pdetz,
                float *poutput) {
        Simulate->calc3d(psrcx, psrcy, psrcz, pdetx, pdety, pdetz, poutput);
        }

    void calc2d(Simulate *Simulate,
                float *psrcx, float *psrcy,
                float *pdetx, float *pdety,
                float *poutput) {
        Simulate->calc2d(psrcx, psrcy, pdetx, pdety, poutput);
        }
    } // extern "C"

Simulate::Simulate(Source *pSource, Detector *pDetector,
                   Phantom* pPhantom, float *pinput) :
    pinput_(pinput),
    pSource_(pSource),
    pDetector_(pDetector),
    det_sizex_(pDetector->sizex),
    det_sizey_(pDetector->sizey),
    pPhantom_(pPhantom),
    obj_sizex_(pPhantom->sizex),
    obj_sizey_(pPhantom->sizey),
    obj_sizez_(pPhantom->sizez),
    obj_pixel_size_(pPhantom->pixel_size) {
    int m;
    for (m = 0; m < obj_sizex_ + 1; m++) {
        xi_.push_back(obj_pixel_size_ * (-float(obj_sizex_) / 2 + m));
        }
    for (m = 0; m < obj_sizey_ + 1; m++) {
        yi_.push_back(obj_pixel_size_ * (-float(obj_sizey_) / 2 + m));
        }
    for (m = 0; m < obj_sizez_ + 1; m++) {
        zi_.push_back(obj_pixel_size_ * (-float(obj_sizez_) / 2 + m));
        }
    }

void Simulate::calc3d(float *psrcx, float *psrcy, float *psrcz,
                      float *pdetx, float *pdety, float *pdetz,
                      float *poutput) {
    int m, n;
    num_pts_ = det_sizex_ * det_sizey_;
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

        xf_ = (xi_[0] - psrcx[m]) / (pdetx[m] - psrcx[m]);
        xl_ = (xi_[obj_sizex_] - psrcx[m]) / (pdetx[m] - psrcx[m]);
        yf_ = (yi_[0] - psrcy[m]) / (pdety[m] - psrcy[m]);
        yl_ = (yi_[obj_sizey_] - psrcy[m]) / (pdety[m] - psrcy[m]);
        zf_ = (zi_[0] - psrcz[m]) / (pdetz[m] - psrcz[m]);
        zl_ = (zi_[obj_sizez_] - psrcz[m]) / (pdetz[m] - psrcz[m]);

        amin_ = fmaxf(fmaxf(0, fminf(xf_, xl_)), fmaxf(fminf(yf_, yl_), fminf(zf_, zl_)));
        amax_ = fminf(fminf(1, fmaxf(xf_, xl_)), fminf(fmaxf(yf_, yl_), fmaxf(zf_, zl_)));

        for (n = 0; n < obj_sizex_ + 1; n++) {
            tmp = (xi_[n] - psrcx[m]) / (pdetx[m] - psrcx[m]);
            if ((tmp >= amin_) && (tmp <= amax_)) {
                ax_.push_back(tmp);
                }
            }
        for (n = 0; n < obj_sizey_ + 1; n++) {
            tmp = (yi_[n] - psrcy[m]) / (pdety[m] - psrcy[m]);
            if ((tmp >= amin_) && (tmp <= amax_)) {
                ay_.push_back(tmp);
                }
            }
        for (n = 0; n < obj_sizez_ + 1; n++) {
            tmp = (zi_[n] - psrcz[m]) / (pdetz[m] - psrcz[m]);
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
                xk_.push_back(psrcx[m] + alpha_[n] * (pdetx[m] - psrcx[m]));
                yk_.push_back(psrcy[m] + alpha_[n] * (pdety[m] - psrcy[m]));
                zk_.push_back(psrcz[m] + alpha_[n] * (pdetz[m] - psrcz[m]));
                }

            for (n = 0; n < alpha_.size() - 1; n++) {
                dist_.push_back(sqrt(pow(xk_[n + 1] - xk_[n], 2) + pow(yk_[n + 1] - yk_[n], 2) + pow(zk_[n + 1] - zk_[n], 2)));
                }

            for (n = 0; n < dist_.size(); n++) {
                indx_ = floor(((xk_[n] + xk_[n + 1]) / 2) / obj_pixel_size_ + float(obj_sizex_) / 2);
                indy_ = floor(((yk_[n] + yk_[n + 1]) / 2) / obj_pixel_size_ + float(obj_sizey_) / 2);
                indz_ = floor(((zk_[n] + zk_[n + 1]) / 2) / obj_pixel_size_ + float(obj_sizez_) / 2);
                ind_out_ = indz_ * (obj_sizex_ * obj_sizey_) + indy_ * obj_sizex_ + indx_ ;
                poutput[m] += pinput_[ind_out_] * dist_[n];
                }
            }
        }
    }


void Simulate::calc2d(float *psrcx, float *psrcy,
                      float *pdetx, float *pdety,
                      float *poutput) {
    int m, n, k;
    for (m = 0; m < det_sizey_; m++) {
        if (!ax_.empty()) { ax_.clear(); }
        if (!ay_.empty()) { ay_.clear(); }
        if (!az_.empty()) { az_.clear(); }
        if (!axy_.empty()) { axy_.clear(); }
        if (!alpha_.empty()) { alpha_.clear(); }
        if (!xk_.empty()) { xk_.clear(); }
        if (!yk_.empty()) { yk_.clear(); }
        if (!zk_.empty()) { zk_.clear(); }
        if (!dist_.empty()) { dist_.clear(); }

        xf_ = (xi_[0] - psrcx[m]) / (pdetx[m] - psrcx[m]);
        xl_ = (xi_[obj_sizex_] - psrcx[m]) / (pdetx[m] - psrcx[m]);
        yf_ = (yi_[0] - psrcy[m]) / (pdety[m] - psrcy[m]);
        yl_ = (yi_[obj_sizey_] - psrcy[m]) / (pdety[m] - psrcy[m]);

        amin_ = fmaxf(0, fmaxf(fminf(xf_, xl_), fminf(yf_, yl_)));
        amax_ = fminf(1, fminf(fmaxf(xf_, xl_), fmaxf(yf_, yl_)));

        for (n = 0; n < obj_sizex_ + 1; n++) {
            tmp = (xi_[n] - psrcx[m]) / (pdetx[m] - psrcx[m]);
            if ((tmp >= amin_) && (tmp <= amax_)) {
                ax_.push_back(tmp);
                }
            }
        for (n = 0; n < obj_sizey_ + 1; n++) {
            tmp = (yi_[n] - psrcy[m]) / (pdety[m] - psrcy[m]);
            if ((tmp >= amin_) && (tmp <= amax_)) {
                ay_.push_back(tmp);
                }
            }
        std::merge(ax_.begin(), ax_.end(),
                   ay_.begin(), ay_.end(),
                   std::back_inserter(alpha_));
        std::sort(alpha_.begin(), alpha_.end());
        alpha_.erase(unique(alpha_.begin(), alpha_.end()), alpha_.end());

        if (alpha_.size() > 1) {
            for (n = 0; n < alpha_.size(); n++) {
                xk_.push_back(psrcx[m] + alpha_[n] * (pdetx[m] - psrcx[m]));
                yk_.push_back(psrcy[m] + alpha_[n] * (pdety[m] - psrcy[m]));
                }

            for (n = 0; n < alpha_.size() - 1; n++) {
                dist_.push_back(sqrt(pow(xk_[n + 1] - xk_[n], 2) + pow(yk_[n + 1] - yk_[n], 2)));
                }

            for (n = 0; n < dist_.size(); n++) {
                indx_ = floor(((xk_[n] + xk_[n + 1]) / 2) / obj_pixel_size_ + float(obj_sizex_) / 2);
                indy_ = floor(((yk_[n] + yk_[n + 1]) / 2) / obj_pixel_size_ + float(obj_sizey_) / 2);
                for (k = 0; k < det_sizex_; k++) {
                    indz_ = k;
                    ind_out_ = indz_ * (obj_sizex_ * obj_sizey_) + indy_ * obj_sizex_ + indx_ ;
                    poutput[m + k * det_sizey_] += pinput_[ind_out_] * dist_[n];
                    }
                }
            }
        }
    }
