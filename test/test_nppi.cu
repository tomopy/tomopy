#include <chrono>
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

//======================================================================================//

#if !defined(START_TIMER)
#    define START_TIMER(var) auto var = std::chrono::system_clock::now()
#endif

//======================================================================================//

#if !defined(REPORT_TIMER)
#    define REPORT_TIMER(start_time, note, counter, total_count)                         \
        {                                                                                \
            auto                          end_time = std::chrono::system_clock::now();   \
            std::chrono::duration<double> elapsed_seconds = end_time - start_time;       \
            printf("[%li]> %-16s :: %8i of %8i... %5.2f seconds\n", 0, note, counter,    \
                   total_count, elapsed_seconds.count());                                \
            start_time = std::chrono::system_clock::now();                               \
        }
#endif

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
print_array(const _Tp* data, int nx, int ny, const std::string& desc, const float& theta)
{
    std::stringstream ss;
    ss << std::setprecision(1) << std::fixed << std::setw(5) << "[theta = " << theta
       << "] " << desc << "\n\n";
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
rotate(float* dst, const float* src, double theta, int nx, int ny, int which,
       int eInterp = INTER_CUBIC, int verbose = 0)
{
    auto getRotationMatrix2D = [&](double m[2][3], double scale) {
        double angle    = theta * (M_PI / 180.0);
        double alpha    = scale * cos(angle);
        double beta     = scale * sin(angle);
        double center_x = (0.5 * nx) - 0.5;
        double center_y = (0.5 * ny) - 0.5;

        if(verbose > 2)
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

    NppiRect roi;
    roi.x      = 0;
    roi.y      = 0;
    roi.width  = nx;
    roi.height = ny;

    int    step = nx * sizeof(float);
    double rot[2][3];
    getRotationMatrix2D(rot, 1.0);

    if(verbose > 1)
        print_array((double*) rot, 3, 2, "rot", theta);

    NppStatus ret;
    if(which % 2 == 0)
        ret = nppiWarpAffine_32f_C1R(src, siz, step, roi, dst, step, roi, rot, eInterp);
    else
        ret = nppiRotate_32f_C1R(src, siz, step, roi, dst, step, roi, theta, rot[0][2],
                                 rot[1][2], eInterp);

    if(ret != NPP_SUCCESS)
        printf("%s returned non-zero NPP status: %i\n", __FUNCTION__, ret);

    cudaStreamSynchronize(0);
}

//============================================================================//

void
usage(int n, double theta, int ntot, int eInterp)
{
    printf("\n\nusage: <size> <theta> <tot-value> <interp-mode>\n");
    printf("\n\t%-15s: %s (default = %i) -- %s\n", "size", "N", n, "array size");
    printf("\n\t%-15s: %s (default = %4.0f) -- %s\n", "theta", "deg", theta,
           "array size");
    printf("\n\t%-15s: %s (default = %i) -- %s\n", "tot-value", "ntot", n,
           "number of values set in array");
    printf("\n\t%-15s: %s (default = %i) -- %s\n", "interp-mode", "MODE", n,
           "mode of interpolation");
    exit(0);
}

//============================================================================//

void
exec(int n, double theta, int eInterp, int nitr, int verbose, int which, float* src,
     float* dst)
{
    if(verbose > 0)
        print_array(src, n, n, "src", 0);

    if(nitr > 1)
        printf("\nALGORITHM = %s\n\n", (which % 2 == 0) ? "warpAffine" : "rotate");

    float  step    = (nitr > 1) ? theta / (nitr - 1) : theta;
    float* gpu_src = malloc_and_memcpy<float>(src, n * n);
    float* gpu_dst = malloc_and_memcpy<float>(dst, n * n);

    START_TIMER(tot_timer);
    START_TIMER(itr_timer);
    for(uint32_t i = 0; i < nitr; ++i)
    {
        float _theta = step * i;

        rotate(gpu_dst, gpu_src, -_theta, n, n, which, eInterp, verbose);

        if(verbose > 0)
        {
            cpu_memcpy(gpu_dst, dst, n * n);
            print_array(dst, n, n, "dst", -_theta);
        }

        rotate(gpu_src, gpu_dst, _theta, n, n, which, eInterp, verbose);

        if(verbose > 0)
        {
            cpu_memcpy(gpu_src, dst, n * n);
            print_array(dst, n, n, "src", _theta);
        }

        if(nitr > 1 && i % ((nitr > 10) ? (nitr / 10) : 1) == 0)
            REPORT_TIMER(itr_timer, "iteration", i, nitr);
    }
    if(nitr > 1)
        REPORT_TIMER(itr_timer, "iteration", nitr, nitr);
    if(nitr > 1)
        REPORT_TIMER(tot_timer, "total_time", 0, 0);

    printf("\n");
    cudaFree(gpu_src);
    cudaFree(gpu_dst);

    cudaDeviceSynchronize();
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
    int    nitr    = 100000;
    int    verbose = 0;

    for(int i = 1; i < argc; ++i)
    {
        if(argv[i] == "-h" || argv[i] == "--help")
            usage(n, theta, ntot, eInterp);
    }

    if(argc > 1)
        n = atoi(argv[1]);
    if(argc > 2)
        theta = atof(argv[2]);
    if(argc > 3)
        ntot = atoi(argv[3]);
    if(argc > 4)
        eInterp = atoi(argv[4]);

    if(eInterp == 0)
        eInterp = INTER_NN;
    else if(eInterp == 1)
        eInterp = INTER_LINEAR;
    else
        eInterp = INTER_CUBIC;

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

    exec(n, theta, eInterp, 1, verbose, 0, src, dst);

    exec(n, theta, eInterp, nitr, verbose, 0, src, dst);
    exec(n, theta, eInterp, nitr, verbose, 1, src, dst);

    return 0;
}
