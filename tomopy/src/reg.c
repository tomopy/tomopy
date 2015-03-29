// Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.

// Copyright 2015. UChicago Argonne, LLC. This software was produced 
// under U.S. Government contract DE-AC02-06CH11357 for Argonne National 
// Laboratory (ANL), which is operated by UChicago Argonne, LLC for the 
// U.S. Department of Energy. The U.S. Government has rights to use, 
// reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR 
// UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR 
// ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is 
// modified to produce derivative works, such modified software should 
// be clearly marked, so as not to confuse it with the version available 
// from ANL.

// Additionally, redistribution and use in source and binary forms, with 
// or without modification, are permitted provided that the following 
// conditions are met:

//     * Redistributions of source code must retain the above copyright 
//       notice, this list of conditions and the following disclaimer. 

//     * Redistributions in binary form must reproduce the above copyright 
//       notice, this list of conditions and the following disclaimer in 
//       the documentation and/or other materials provided with the 
//       distribution. 

//     * Neither the name of UChicago Argonne, LLC, Argonne National 
//       Laboratory, ANL, the U.S. Government, nor the names of its 
//       contributors may be used to endorse or promote products derived 
//       from this software without specific prior written permission. 

// THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS 
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago 
// Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.

#include "utils.h"

void reg_term(
        float *model, 
        int num_slices, int num_grids_x, int num_grids_y,
        float beta, float delta,
        float *reg)
{
    int k, n, m, q;
    int ind0, indg[8];
    float totalwg;
    float wg[8];
    float mg[8];
    float rg[8];
    float gammag[8];

    // Weights for inner neighborhoods.
    totalwg = 4+4/sqrt(2);
    wg[0] = 1/totalwg;
    wg[1] = 1/totalwg;
    wg[2] = 1/totalwg;
    wg[3] = 1/totalwg;
    wg[4] = 1/sqrt(2)/totalwg;
    wg[5] = 1/sqrt(2)/totalwg;
    wg[6] = 1/sqrt(2)/totalwg;
    wg[7] = 1/sqrt(2)/totalwg;
        
    // (inner region)
    for (k = 0; k < num_slices; k++) {
        for (n = 1; n < num_grids_x-1; n++) {
            for (m = 1; m < num_grids_y-1; m++) {
                ind0 = m + n*num_grids_y + k*num_grids_x*num_grids_y;

                indg[0] = ind0+1;
                indg[1] = ind0-1;
                indg[2] = ind0+num_grids_y;
                indg[3] = ind0-num_grids_y;
                indg[4] = ind0+num_grids_y+1; 
                indg[5] = ind0+num_grids_y-1;
                indg[6] = ind0-num_grids_y+1;
                indg[7] = ind0-num_grids_y-1;
                
                for (q = 0; q < 8; q++) {
                    rg[q] = model[ind0]-model[indg[q]];
                    mg[q] = fabs(rg[q]/delta);
                    gammag[q] = delta*delta*(mg[q]-log(1+mg[q]));
                    reg[ind0] += wg[q]*gammag[q];
                }
            }
        }
    }
}