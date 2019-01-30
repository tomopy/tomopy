    1  python setup.py install && ./pyctest_tomopy_phantom.py -i 1
    2  time TOMOPY_USE_C_SIRT=0 ./pyctest_tomopy_phantom.py -i 35
    3  time TOMOPY_USE_C_SIRT=1 ./pyctest_tomopy_phantom.py -i 35
    4  ? 145.7/12.3
    5  export TOMOPY_USE_CPU=1
    6  python setup.py install -- -DTOMOPY_USE_PTL=OFF && TOMOPY_USE_C_SIRT=0 ./pyctest_tomopy_phantom.py -i 1
    7  ssed -i '/PRINT_HERE/d' ../src/cxx/sirt.cc 
    8  sed -i '/PRINT_HERE/d' ../src/cxx/sirt.cc 
    9  python setup.py install -- -DTOMOPY_USE_PTL=OM && TOMOPY_USE_C_SIRT=0 ./pyctest_tomopy_phantom.py -i 1
   10  python setup.py install -- -DTOMOPY_USE_PTL=ON && TOMOPY_USE_C_SIRT=0 ./pyctest_tomopy_phantom.py -i 1
   11  cd ..
   12  emacs src/PTL/cmake/Modules/MacroUtilities.cmake
   13* emacs src/PTL/cmake/Modules/Opt
   14  emacs src/PTL/cmake/Modules/BuildSettings.cmake 
   15  emacs +897 src/PTL/cmake/Modules/MacroUtilities.cmake
   16  git checkout cmake/Modules/BuildSettings.cmake
   17  python setup.py build -- -DTOMOPY_USE_PTL=ON
   18  cd src/PTL/
   19  git commit -m "Fixed compile error if CXX flags is empty"
   20  cd ../..
   21  python setup.py build -- -DTOMOPY_USE_PTL=ON -- format format-ptl
   22  python setup.py clean
   23  module load pgi
   24  python setup.py build -- -DTOMOPY_USE_PTL=ON 
   25  python setup.py build -- -DTOMOPY_USE_PTL=ON -DTOMOPY_USE_PYBIND11=ON
   26  python setup.py build -- -DTOMOPY_USE_PTL=OFF -DTOMOPY_USE_PYBIND11=ON
   27  python setup.py build -- -DTOMOPY_USE_TIMEMORY=ON
   28  git status
   29  git add .
   30  git commit -m "Fixed PGI warnings"
   31  rm -rf _skbuild/
   32  module unload pgi/18.10 
   33  cd benchmarking/
   34  python setup.py install && CUDA_BLOCK_SIZE=64 ./pyctest_tomopy_rec.py /home/globus/tomo_00001/tomo_00001.h5 -o tomo_00001_sirt/block_64 -i 10 && CUDA_BLOCK_SIZE=32 ./pyctest_tomopy_rec.py /home/globus/tomo_00001/tomo_00001.h5 -o tomo_00001_sirt/block_32 -i 10
   35  env | grep TOMO
   36  unset TOMOPY_USE_CPU 
   37  python setup.py install && ./pyctest_tomopy_rec.py /home/globus/tomo_00001/tomo_00001.h5 -o tomo_00001_sirt/block_128 -i 10 && CUDA_BLOCK_SIZE=64 ./pyctest_tomopy_rec.py /home/globus/tomo_00001/tomo_00001.h5 -o tomo_00001_sirt/block_64 -i 10 && CUDA_BLOCK_SIZE=32 ./pyctest_tomopy_rec.py /home/globus/tomo_00001/tomo_00001.h5 -o tomo_00001_sirt/block_32 -i 10
   38  apropos nsight
   39  apt-get update
   40  apt-cache search nsight
   41  dpkg -L nvidia-nsight
   42  apt-get install nvidia-nsight
   43  apt-get install cuda-visual-tools-10-0 cuda-nsight-compute-10-0 cuda-nsight-10-0
   44  nsight --help
   45  apt-file update
   46  nvprof --help
   47  dpkg -L cuda-nsight-compute-10-0
   48  dpkg -L cuda-nsight-compute-10-0 | grep bin
   49  export PATH=/usr/local/cuda-10.0/NsightCompute-1.0:${PATH}
   50  nv-nsight-cu --help
   51  nv-nsight-cu-cli --help
   52  python setup.py install && nv-nsight-cu-cli -o block_128 --nvtx ./pyctest_tomopy_rec.py /home/globus/tomo_00001/tomo_00001.h5 -o tomo_00001_sirt/block_128 -i 10 && CUDA_BLOCK_SIZE=32 nv-nsight-cu-cli -o block_32 --nvtx ./pyctest_tomopy_rec.py /home/globus/tomo_00001/tomo_00001.h5 -o tomo_00001_sirt/block_32 -i 10
   53  python setup.py install && nv-nsight-cu-cli -o block_128 --nvtx python ./pyctest_tomopy_rec.py /home/globus/tomo_00001/tomo_00001.h5 -o tomo_00001_sirt/block_128 -i 10 && CUDA_BLOCK_SIZE=32 nv-nsight-cu-cli -o block_32 --nvtx python ./pyctest_tomopy_rec.py /home/globus/tomo_00001/tomo_00001.h5 -o tomo_00001_sirt/block_32 -i 10
   54  python setup.py install && nv-nsight-cu-cli -o block_128 --nvtx $(which python) ./pyctest_tomopy_rec.py /home/globus/tomo_00001/tomo_00001.h5 -o tomo_00001_sirt/block_128 -i 10 && CUDA_BLOCK_SIZE=32 nv-nsight-cu-cli -o block_32 --nvtx $(which python) ./pyctest_tomopy_rec.py /home/globus/tomo_00001/tomo_00001.h5 -o tomo_00001_sirt/block_32 -i 10
   55  ls -t
   56  python setup.py install && nv-nsight-cu-cli -o block_128 --nvtx $(which python) ./pyctest_tomopy_rec.py /home/globus/tomo_00001/tomo_00001.h5 -o tomo_00001_sirt/block_128 -i 1 && CUDA_BLOCK_SIZE=32 nv-nsight-cu-cli -o block_32 --nvtx $(which python) ./pyctest_tomopy_rec.py /home/globus/tomo_00001/tomo_00001.h5 -o tomo_00001_sirt/block_32 -i 1
   57  rm block_32.nsight-cuprof-report block_128.nsight-cuprof-report 
   58  python setup.py install && nv-nsight-cu-cli -o block_128 --nvtx -c 100 $(which python) ./pyctest_tomopy_rec.py /home/globus/tomo_00001/tomo_00001.h5 -o tomo_00001_sirt/block_128 -i 1 && CUDA_BLOCK_SIZE=32 nv-nsight-cu-cli -o block_32 --nvtx -c 100 $(which python) ./pyctest_tomopy_rec.py /home/globus/tomo_00001/tomo_00001.h5 -o tomo_00001_sirt/block_32 -i 1
   59  locate libimf
   60  python setup.py install
   61  python setup.py install && CUDA_BLOCK_SIZE=128 python ./pyctest_tomopy_rec.py /home/globus/tomo_00001/tomo_00001.h5 -o tomo_00001_sirt/block_128 -i 10 && CUDA_BLOCK_SIZE=32 python ./pyctest_tomopy_rec.py /home/globus/tomo_00001/tomo_00001.h5 -o tomo_00001_sirt/block_32 -i 10
   62  ? 3679.787/52.510
   63  export TOMOPY_NUM_THREADS=4
   64  python setup.py install && nv-nsight-cu-cli -o block_128.1 --nvtx -c 100 $(which python) ./pyctest_tomopy_rec.py /home/globus/tomo_00001/tomo_00001.h5 -o tomo_00001_sirt/block_128 -i 1 && CUDA_BLOCK_SIZE=32 nv-nsight-cu-cli -o block_32.1 --nvtx -c 100 $(which python) ./pyctest_tomopy_rec.py /home/globus/tomo_00001/tomo_00001.h5 -o tomo_00001_sirt/block_32 -i 1
   65  export TOMOPY_NUM_THREADS=1
   66  python setup.py install && nv-nsight-cu-cli -o block_128.2 --nvtx -c 100 $(which python) ./pyctest_tomopy_rec.py /home/globus/tomo_00001/tomo_00001.h5 -o tomo_00001_sirt/block_128 -i 1 && CUDA_BLOCK_SIZE=32 nv-nsight-cu-cli -o block_32.2 --nvtx -c 100 $(which python) ./pyctest_tomopy_rec.py /home/globus/tomo_00001/tomo_00001.h5 -o tomo_00001_sirt/block_32 -i 1
   67  export TOMOPY_NUM_THREADS=8
   68* python setup.py install && nv-nsight-cu-cli -o block_128.3 --nvtx -c 100 $(which python) ./pyctest_tomopy_rec.py /home/globus/tomo_00001/tomo_00001.h5 -o tomo_00001_sirt/block_128 -i 1 && CUDA_BLOCK_SIZE=32 nv-nsight-cu-cli -o block_32.3 --nvtx -c 100 $(which python) ./pyctest_tomopy_rec.py /home/globus/tomo_00001/tomo_00001.h5 -o tomo_00001_sirt/block_32 -i 
   69  N=4; python setup.py install && nv-nsight-cu-cli -o block_128.${N} --nvtx -c 100 $(which python) ./pyctest_tomopy_rec.py /home/globus/tomo_00001/tomo_00001.h5 -o tomo_00001_sirt/block_128 -i 1 && CUDA_BLOCK_SIZE=32 nv-nsight-cu-cli -o block_32.${N} --nvtx -c 100 $(which python) ./pyctest_tomopy_rec.py /home/globus/tomo_00001/tomo_00001.h5 -o tomo_00001_sirt/block_32 -i 1
   70  history > .bash_history
