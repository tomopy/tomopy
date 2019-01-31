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
    1  conda activate tomopy
    2  python setup.py install
    3  ./pyctest_tomopy_phantom.py -i 20
    4  python setup.py install && TOMOPY_NUM_THREADS=4 ./pyctest_tomopy_rec.py -i 10 -o tomo_00001_sirt/block_32 /home/globus/tomo_00001/tomo_00001.h5 
    5  python setup.py install && TOMOPY_NUM_THREADS=4 ./pyctest_tomopy_rec.py -i 10 -o tomo_00001_sirt/block_128 /home/globus/tomo_00001/tomo_00001.h5 
    6  python setup.py install && ./pyctest_tomopy_rec.py -i 10 -o tomo_00001_sirt/block_128 /home/globus/tomo_00001/tomo_00001.h5 
    7  export TOMOPY_NUM_THREADS=4
    8  python setup.py install && ./pyctest_tomopy_rec.py -i 10 -o tomo_00001_sirt/block_128_thread_4 /home/globus/tomo_00001/tomo_00001.h5 
    9  python setup.py install && ./pyctest_tomopy_rec.py -i 10 -o tomo_00001_sirt/block_128_thread_${TOMOPY_NUM_THREADS} /home/globus/tomo_00001/tomo_00001.h5 
   10  cat ../.bash_history 
   11  echo $PATH
   12  ls /usr/local/cuda-10.0/
   13  export PATH=${PATH}:/usr/local/cuda-10.0/NsightCompute-1.0
   14  nv-nsight-cu-cli --help
   15  tail ../.bash_history 
   16  ./profile.sh 
   17  cat profile.sh 
   18  mv block_128.0.nsight-cuprof-report block_128_thread_8.0.nsight-cuprof-report
   19  ./profile.sh 1
   20  ./profile.sh 1 10
   21  ./profile.sh 2 10
   22  ./profile.sh 3 10
   23  ./profile.sh 4 10
   24  cd ..
   25  python setup.py --help
   26  python setup.py --disable-nvtx install
   27  cd benchmarking/
   28  emacs profile.sh 
   29  ls -t
   30* ./profile-n
   31  export CUDA_BLOCK_SIZE=256
   32  ./profile-nvprof.sh 5 10
   33  cat profile-nvprof.sh
   34  export TOMOPY_NUM_THREADS=8
   35  export NITER=10
   36  unset NITER 
   37  export NITR=10
   38  export CUDA_BLOCK_SIZE=32
   39  ./pyctest_tomopy_rec.py        -o tomo_00001_sirt/block_${CUDA_BLOCK_SIZE}_threads_${TOMOPY_NUM_THREADS}        -i ${NITR}        /home/globus/tomo_00001/tomo_00001.h5
   40  export CUDA_BLOCK_SIZE=64
   41  export CUDA_BLOCK_SIZE=128
   42  python setup.py --enable-nvtx install && ./pyctest_tomopy_rec.py -o tomo_00001_sirt/block_${CUDA_BLOCK_SIZE}_threads_${TOMOPY_NUM_THREADS} -i ${NITR} /home/globus/tomo_00001/tomo_00001.h5
   43  ython setup.py --enable-nvtx install && ./pyctest_tomopy_rec.py -o tomo_00001_sirt/block_${CUDA_BLOCK_SIZE}_threads_${TOMOPY_NUM_THREADS} -i ${NITR} /home/globus/tomo_00001/tomo_00001.h5
   44  export TOMOPY_NUM_THREADS=12
   45  ? 3359.400/39.97
   46  ? 3679.787/39.97
   47  TIME=18.81; echo -e "Speed-up vs. Haswell: $(calc 3679.787/${TIME})"; echo -e "Speed-up vs. Edison: $(calc 3359.400/${TIME})" 
   48  TIME=18.81; echo -e "  Speed-up vs. Haswell: $(calc 3679.787/${TIME})"; echo -e "   Speed-up vs. Edison: $(calc 3359.400/${TIME})" 
   49  TIME=18.81; echo -e "  Speed-up vs. Haswell:\t $(calc 3679.787/${TIME})"; echo -e "   Speed-up vs. Edison:\t $(calc 3359.400/${TIME})" 
   50  TIME=15.29; echo -e "  Speed-up vs. Haswell:\t $(calc 3679.787/${TIME})"; echo -e "   Speed-up vs. Edison:\t $(calc 3359.400/${TIME})" 
   51  python setup.py --disable-nvtx install && ./pyctest_tomopy_rec.py -o tomo_00001_sirt/block_${CUDA_BLOCK_SIZE}_threads_${TOMOPY_NUM_THREADS} -i ${NITR} /home/globus/tomo_00001/tomo_00001.h5
   52  cp tomo_00001_sirt/block_128_threads_12/rec_slice/recon_sirt_0.tif ~/cubic.tif
   53  cp tomo_00001_sirt/block_128_threads_12/rec_slice/recon_sirt_0.tif ~/linear.tif
   54  cp tomo_00001_sirt/block_128_threads_12/rec_slice/recon_sirt_0.tif ~/nn.tif
   55  mv ~/*.tif ./
   56  export TOMOPY_USE_C_SIRT=1
   57  python setup.py --disable-nvtx install && ./pyctest_tomopy_rec.py -o tomo_00001_sirt/block_${CUDA_BLOCK_SIZE}_threads_${TOMOPY_NUM_THREADS} -i ${NITR} /home/globus/tomo_00001/tomo_00001.h5 -f tif
   58  cp tomo_00001_sirt/block_128_threads_12/rec_slice/recon_sirt_0.tif ~/c.tif
   59  cp tomo_00001_sirt/block_128_threads_12/rec_slice/recon_sirt_0.tif c.tif
   60  history >> ../.bash_history 
