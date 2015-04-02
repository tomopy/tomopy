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

void art(
        float *data, float* theta, float center,
        int num_projs, int num_slices, int num_pixels,
        int num_grids_x, int num_grids_y, int num_iters,
        float *model) // Recon is reductiom object
{
 //    float* gridx = (float *)malloc((num_grids_x+1)*sizeof(float));
 //    float* gridy = (float *)malloc((num_grids_y+1)*sizeof(float));
 //    float* coordx = (float *)malloc((num_grids_y+1)*sizeof(float));
 //    float* coordy = (float *)malloc((num_grids_x+1)*sizeof(float));
 //    float* ax = (float *)malloc((num_grids_x+num_grids_y)*sizeof(float));
 //    float* ay = (float *)malloc((num_grids_x+num_grids_y)*sizeof(float));
 //    float* bx = (float *)malloc((num_grids_x+num_grids_y)*sizeof(float));
 //    float* by = (float *)malloc((num_grids_x+num_grids_y)*sizeof(float));
 //    float* coorx = (float *)malloc((num_grids_x+num_grids_y)*sizeof(float));
 //    float* coory = (float *)malloc((num_grids_x+num_grids_y)*sizeof(float));
 //    float* dist = (float *)malloc((num_grids_x+num_grids_y)*sizeof(float));
 //    int* indi = (int *)malloc((num_grids_x+num_grids_y)*sizeof(int));

 //    assert(coordx != NULL && coordy != NULL &&
 //        ax != NULL && ay != NULL && by != NULL && bx != NULL &&
 //        coorx != NULL && coory != NULL && dist != NULL && indi != NULL);

 //    int s, p, c;
 //    int i, n;
 //    int quadrant;
 //    float theta_p;
 //    float mov;
 //    float sin_p, cos_p;
 //    float xi, yi;
 //    int asize, bsize, csize;
 //    float* simdata;
 //    float upd;
 //    int index_model, index_data;

 //    preprocessing(num_grids_x, num_grids_y, num_pixels, center, 
 //        &mov, gridx, gridy); // Outputs: mov, gridx, gridy

 //    for (i=0; i<num_iters; i++) {
	//     printf("iteration : %i\n", i+1);

	//     simdata = (float *)calloc((num_projs*num_slices*num_pixels), sizeof(float));

	//     // For each slice
	//     for (s=0; s<num_slices; s++) {

	//         // For each projection angle 
	//         for (p=0; p<num_projs; p++) {

	//             // Calculate the sin and cos values 
	//             // of the projection angle and find
	//             // at which quadrant on the cartesian grid.
	//             theta_p = fmod(theta[p], 2*M_PI);
	//             quadrant = calc_quadrant(theta_p);
	//             sin_p = sinf(theta_p);
	//             cos_p = cosf(theta_p);

	//             for (c=0; c<num_pixels; c++) {
	//                 // Calculate coordinates
	//                 xi = -1e6;
	//                 yi = -(num_pixels-1)/2.0+c+mov;
	//                 calc_coords(num_grids_x, num_grids_y, xi, yi, sin_p, cos_p, gridx, gridy, 
	//                         coordx, coordy); // Outputs: coordx, coordy

	//                 // Merge the (coordx, gridy) and (gridx, coordy)
	//                 trim_coords(num_grids_x, num_grids_y, coordx, coordy, gridx, gridy, 
	//                         &asize, ax, ay, &bsize, bx, by); // Outputs: asize and after

	//                 // Sort the array of intersection points (ax, ay) and (bx, by)
	//                 // The new sorted intersection points are stored in (coorx, coory).
	//                 // Total number of points are csize.
	//                 sort_intersections(quadrant, asize, ax, ay, bsize, bx, by, 
	//                     &csize, coorx, coory); // Outputs: csize, coorx, coory

	//                 // Calculate the distances (dist) between the 
	//                 // intersection points (coorx, coory). Find 
	//                 // the indices of the pixels on the reconstruction grid (ind_recon).
	//                 calc_dist(num_grids_x, num_grids_y, csize, coorx, coory, 
	//                     indi, dist); // Outputs: indi, dist 

	//                 // Calculate simdata 
	//                 calc_simdata(p, s, c, num_grids_x, num_grids_y, 
	//                 	num_slices, num_pixels, csize, indi, dist, model,
	//                     simdata); // Output: simdata


	//                 // Calculate dist*dist
	//                 float sum_dist2 = 0.0;
	//                 for (n=0; n<csize-1; n++) {
	//        		 		sum_dist2 += dist[n]*dist[n];
	//     			}

	//                 // Update
	//                 index_model = s*num_grids_x*num_grids_y;
	//                 index_data = c+s*num_pixels+p*num_slices*num_pixels;
	//                 if (sum_dist2 != 0.0) {
	// 	                upd = (data[index_data]-simdata[index_data])/sum_dist2;
	// 	                for (n=0; n<csize-1; n++) {
	// 						model[indi[n]+index_model] += upd*dist[n];
	// 					}
	// 				}
	//             }
	//         }
	//     }

	//     free(simdata);
	// }

 //    free(gridx);
 //    free(gridy);
 //    free(coordx);
 //    free(coordy);
 //    free(ax);
 //    free(ay);
 //    free(bx);
 //    free(by);
 //    free(coorx);
 //    free(coory);
 //    free(dist);
 //    free(indi);
}