#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <iostream>
#include <iomanip>

using namespace cv;
using namespace std;

const char* source_label     = "Source image";
const char* forward_label    = "Forward Rotate";
const char* backward_label   = "Backward Rotate";
const char* difference_label = "Difference image";
const char* zoom_diff_label  = "Zoomed Difference";

#define PRINT_HERE(extra) printf("[%s] @ %i :: %s\n", __FUNCTION__, __LINE__, extra)

//============================================================================//

Mat
cxx_affine_transform(const Mat& warp_src, float theta, int nx, int ny, float scale)
{
    Mat   warp_dst = Mat::zeros(nx * 2, ny * 2, warp_src.type());
    float cx       = 0.5f * ny + ((ny % 2 == 0) ? 0.5f : 0.0f);
    float cy       = 0.5f * nx + ((nx % 2 == 0) ? 0.5f : 0.0f);
    Point center   = Point(cx, cy);
    Mat   rot      = getRotationMatrix2D(center, theta, scale);
    std::cout.precision(3);
    std::cout << std::fixed;
    std::cout << "theta = " << theta
              //<< ", rot = \n" << rot
              << std::endl;
    for(int i = 0; i < rot.rows; ++i)
    {
        for(int j = 0; j < rot.cols; ++j)
        {
            std::cout << std::setw(8) << std::setprecision(3) << std::fixed
                      << rot.at<float>(i, j) << " ";
        }
        std::cout << std::endl;
    }
    warpAffine(warp_src, warp_dst, rot, warp_src.size(), INTER_CUBIC);
    return warp_dst;
}

//============================================================================//

void
plot(const char* label, const Mat& warp_mat)
{
    namedWindow(label, WINDOW_AUTOSIZE);
    imshow(label, warp_mat);
}

//============================================================================//

int
main(int argc, char** argv)
{
    string file  = "tomopy/data/cameraman.tif";
    float  theta = -45.0f;
    float  scale = 1.0f;

    if(argc > 1)
        file = argv[1];
    if(argc > 2)
        theta = atof(argv[2]);
    if(argc > 3)
        scale = atof(argv[3]);

    // Mat rot_mat(2, 3, CV_32FC1);
    PRINT_HERE("");

    Mat  warp_src  = imread(file);
    auto nx        = warp_src.rows;
    auto ny        = warp_src.cols;
    Mat  warp_forw = cxx_affine_transform(warp_src, -theta, nx, ny, scale);
    Mat  warp_back = cxx_affine_transform(warp_forw, theta, nx, ny, scale);
    Mat  warp_diff = warp_src - warp_back;
    Mat  warp_zoom = cxx_affine_transform(warp_diff, 0.0f, nx, ny,
                                         1.0f / cosf(theta * (M_PI / 180.f)));

    plot(zoom_diff_label, warp_zoom);
    plot(difference_label, warp_diff);
    plot(backward_label, warp_back);
    plot(forward_label, warp_forw);
    plot(source_label, warp_src);

    waitKey(0);

    return 0;
}
