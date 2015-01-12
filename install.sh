python install.py . --boost --fftw
export LD_LIBRARY_PATH=./lib
export C_INCLUDE_PATH=./include
python setup.py build_ext --force -b .
python setup.py install

echo "LD_LIBRARY_PATH = $LD_LIBRARY_PATH"
echo "C_INCLUDE_PATH = $C_INCLUDE_PATH"
echo "PYTHONPATH = $PYTHONPATH"
python -c "import tomopy"
