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

#include <stdio.h>
#include <math.h>

#ifdef WIN32
#define DLL __declspec(dllexport)
#else
#define DLL 
#endif

DLL void downsample2d(float* data, int num_projections,
                  int num_slices, int num_pixels,
                  int level, float* downsampled_data) {

    int m, n, k, i, p, iproj, ind;
    int binsize;
    
    binsize = pow(2, level);
    
    num_pixels /= binsize;

    for (m = 0, ind = 0; m < num_projections; m++) {
        iproj = m * (num_pixels * num_slices);
        
    for (n = 0; n < num_slices; n++) {
        i = iproj + n * num_pixels;

            for (k = 0; k < num_pixels; k++) {
                        
                for (p = 0; p < binsize; p++) {
                    downsampled_data[i+k] += data[ind]/binsize;
                    ind++;
                }
            }
        }
    }
}

DLL void downsample3d(float* data, int num_projections,
                  int num_slices, int num_pixels,
                  int level, float* downsampled_data) {

    int m, n, k, i, p, q, iproj, ind;
    int binsize, binsize2;
    
    binsize = pow(2, level);
    binsize2 = binsize * binsize;
    
    num_slices /= binsize;
    num_pixels /= binsize;

    for (m = 0, ind = 0; m < num_projections; m++) {
        iproj = m * (num_pixels * num_slices);

        for (n = 0; n < num_slices; n++) {
            i = iproj + n * num_pixels;

            for (q = 0; q < binsize; q++) {
                for (k = 0; k < num_pixels; k++) { 

                    for (p = 0; p < binsize; p++) {
                        downsampled_data[i+k] += data[ind]/binsize2;
                        ind++;
                    }
                }
            }
        }
    }
}