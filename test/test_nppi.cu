#include <cmath>
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iomanip>
#include <iostream>
#include <npp.h>
#include <nppi.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector_types.h>

#define PRINT_HERE(extra) printf("[%s] @ %i :: %s\n", __FUNCTION__, __LINE__, extra)

#define INTER_NN NPPI_INTER_NN
#define INTER_LINEAR NPPI_INTER_LINEAR
#define INTER_CUBIC NPPI_INTER_CUBIC

//============================================================================//

template <typename _Tp>
_Tp*
malloc_and_memcpy(const _Tp* _cpu, uintmax_t size)
{
    _Tp* _gpu;
    cudaMalloc((void**) &_gpu, size * sizeof(_Tp));
    cudaMemcpy(_gpu, _cpu, size * sizeof(_Tp), cudaMemcpyHostToDevice);
    return _gpu;
}

//============================================================================//

template <typename _Tp>
void
cpu_memcpy(const _Tp* _gpu, _Tp* _cpu, uintmax_t size)
{
    cudaMemcpy(_cpu, _gpu, size * sizeof(_Tp), cudaMemcpyDeviceToHost);
}

//============================================================================//

template <typename _Tp>
void
print_array(const _Tp* data, int nx, int ny, const std::string& desc)
{
    std::stringstream ss;
    ss << desc << "\n\n";
    ss << std::fixed;
    ss.precision(3);
    for(int j = 0; j < ny; ++j)
    {
        ss << "  ";
        for(int i = 0; i < nx; ++i)
        {
            ss << std::setw(8) << data[j * nx + i] << " ";
        }
        ss << std::endl;
    }
    std::cout << ss.str() << std::endl;
}

//============================================================================//

void
rotate(float* dst, const float* src, double theta, int nx, int ny,
       int eInterp = INTER_CUBIC)
{
    eInterp %= 4;

    auto getRotationMatrix2D = [&](double m[2][3], double scale) {
        double angle    = theta * (M_PI / 180.0);
        double alpha    = scale * cos(angle);
        double beta     = scale * sin(angle);
        double center_x = (0.5 * nx) - 0.5;
        double center_y = (0.5 * ny) - 0.5;
        printf("center = [%8.3f, %8.3f]\n", center_x, center_y);

        m[0][0] = alpha;
        m[0][1] = beta;
        m[0][2] = (1.0 - alpha) * center_x - beta * center_y;
        m[1][0] = -beta;
        m[1][1] = alpha;
        m[1][2] = beta * center_x + (1.0 - alpha) * center_y;
    };

    NppiSize siz;
    siz.width  = nx;
    siz.height = ny;

    NppiRect src_roi;
    src_roi.x      = 0;
    src_roi.y      = 0;
    src_roi.width  = nx;
    src_roi.height = ny;

    NppiRect dst_roi;
    dst_roi.x      = 0;
    dst_roi.y      = 0;
    dst_roi.width  = nx;
    dst_roi.height = ny;

    int    step = nx * sizeof(float);
    double rot[2][3];
    getRotationMatrix2D(rot, 1.0);

    printf("theta = %5.1f\n", theta);
    print_array((double*) rot, 3, 2, "rot");

    NppStatus ret = nppiRotate_32f_C1R(src, siz, step, src_roi, dst, step, dst_roi, theta,
                                       rot[0][2], rot[1][2], eInterp);

    if(ret != NPP_SUCCESS)
        printf("%s returned non-zero NPP status: %i\n", __FUNCTION__, ret);
}

//============================================================================//

int
main(int argc, char** argv)
{
    printf("\n");
    int    n       = 6;
    double theta   = 180.0;
    int    ntot    = 2;
    int    eInterp = INTER_CUBIC;

    if(argc > 1)
        n = atoi(argv[1]);
    if(argc > 2)
        theta = atof(argv[2]);
    if(argc > 3)
        ntot = atoi(argv[3]);
    if(argc > 4)
        eInterp = atoi(argv[4]) % 4;

    switch(eInterp)
    {
        case INTER_NN: printf("Interpolation = Nearest Neighbor\n\n"); break;
        case INTER_LINEAR: printf("Interpolation = Linear\n\n"); break;
        case INTER_CUBIC: printf("Interpolation = Cubic\n\n"); break;
        default:
            printf("Interpolation = Unknown (%i). Setting to CUBIC...\n\n", eInterp);
            eInterp = INTER_CUBIC;
            break;
    }

    int    nset = 0;
    float* src  = new float[n * n];
    float* dst  = new float[n * n];
    for(int j = 0; j < n; ++j)
        for(int i = 0; i < n; ++i)
        {
            int idx = j * n + i;
            if((i == (n / 2 - 2 + n % 2) || i == (n / 2 + 1)) &&
               (j == (n / 2 - 2 + n % 2) || j == (n / 2 + 1)) && nset < ntot)
            {
                src[idx] = -1.0f * static_cast<float>(nset + 1);
                ++nset;
            }
            else
                src[idx] = 0.0f;
            dst[idx] = 0.0f;
        }

    print_array(src, n, n, "src");

    float* gpu_src = malloc_and_memcpy<float>(src, n * n);
    float* gpu_dst = malloc_and_memcpy<float>(dst, n * n);

    rotate(gpu_dst, gpu_src, -theta, n, n, eInterp);
    cpu_memcpy(gpu_dst, dst, n * n);
    print_array(dst, n, n, "dst");

    rotate(gpu_src, gpu_dst, theta, n, n, eInterp);
    cpu_memcpy(gpu_src, dst, n * n);
    print_array(dst, n, n, "src");

    return 0;
}
