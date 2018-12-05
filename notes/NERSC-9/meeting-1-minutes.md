# NERSC-9 Application Transition COE call

## Meeting #1

### Date: November 28, 2018

### Time: 1:00 PM (Pacific)

- *Introductions*

- *NERSC or code developers describe code generally?*
    - *What algorithms used?*
        - The hack-a-thon will focus on creating a GPU version of the SIRT algorithm
        - A CUDA version of the ART algorithm already exists (with poor performance) in `^/src/cxx/art.cc`
    - *What language(s) is the code written?*
        - TomoPy is C with a thin layer of Python code accessing the C via ctypes
        - TomoPy also supports C++ which can be accessed from Python by either:
            - redirection through C, i.e. C function calls `extern "C"` C++ function (current method)
            - Using PyBind11 (preliminary implementation)
    - *What parallel programming models currently in use?*
        - TomoPy starts threads at the Python level with `concurrent.futures`
        - TomoPy does not currently support MPI
        - TomoPy has a secondary MT tasking system using a package called PTL
            - Goal is to eventually handle thread creation/destruction/workload-balancing in C++ with this package instead of Python

- *What is the scope of the code that will be used for the hackathon?*
    - Full application

- *What is the state of GPU readiness?*
    - TomoPy has some preliminary CUDA implementations that are not optimal
        - The first implementation did some work on the GPU and some on the CPU but the memory transfer overhead was a significant problem
        - The second implementation eliminated almost all the memory transfer overhead but created a new problem:
            - The first implementation had two serial algorithms that ran well on the CPU, a non-traditional sort/filter function and a trim function
            - This second implementation required the kernel launch of these functions to be one block and one thread
            - Thus, while there was a performance gain from the first implementation to the second, the sort/filter and trim algorithms went from ~40% of runtime on CPU to ~99% of runtime on GPU and the performance was still about an order of magnitude slower than CPU-only version
    - A new approach to solving the problem is under-development
        - By applying a coordinate transform to the data before the bulk of the calculations and a back-transform after the calculations, the sort/filter and trimming would no longer be necessary
        - From the CPU perspective, this transform has been considered an unnecessary computational cost but it is believed that this will not be the case from the GPU perspective since the GPU excels at matrix multiplications

- *What GPU programming models are of interest?*
    - The original endeavor attempted separate GPU programming models with CUDA, OpenACC, and OpenMP
        - This path was conceived as a way to create a study/analysis of performance vs. implementation complexity among the models
        - TomoPy iterative algorithms share a common set of 4-5 functions and there are, generally, only minor variations among the algorithms which made the idea seem feasible
    - For this hack-a-thon, TomoPy would like to only address CUDA and OpenACC
        - With OpenACC working, it should be *relatively* easy to add OpenMP later and complete the study

- *What are developers’ objectives for the hackathon?*
    - It can be assumed that the coordinate transform approach will be available for the hack-a-thon
    - Objective 1: Fine-tuning OpenACC and CUDA of the coordinate transform approach
    - Objective 2: Expand experience and knowledge about using profiling tools
        - Developers have general experience using NVIDIA Visual Profiler and NVIDIA Nsight Systems
        - TomoPy, for example, already uses the NVTX API for labeling profiling sections in Nsight
    - Objective 3: Deep-dive into profiling output

- *Identify a test problem to focus on for the hackathon*
    - TomoPy has a built-in Python script capable of running problems with various sizes and runtimes and one will be selected, e.g.
        - `./pyctest_tomopy.py --algorithms sirt --phantom-size 256 --num-iter 10 --phantoms baboon --pyctest-stages Test` is a ~15 second test
        - `./pyctest_tomopy.py --algorithms sirt --phantom-size 512 --num-iter 25 --phantoms baboon --pyctest-stages Test` is a ~90 second test on a larger dataset with more iterations

- *Arrange source code and inputs file access for Cray and NVIDIA (email + tarball, GitHub, etc.)*
    - Code is accessible on GitHub: [github.com/jrmadsen/tomopy](https://github.com/jrmadsen/tomopy) in the `gpu-devel` branch
    - `git clone -b gpu-devel https://github.com/jrmadsen/tomopy.git`
    - `conda env create -f envs/tomopy-gpu-dev.yml`
    - `source activate tomopy-gpu-dev`
    - Build explicitly with CUDA:
        - `python setup.py --enable-cuda install`
    - Build explicitly with OpenACC:
        - `python setup.py --enable-openacc install`
    - Test:
        - `nosetests`

- *Other topics:*
    - File PGI bug reports for PGI compilers + C++11
