Installing TomoPy
=================

TomoPy requires HDF5, Boost and FFTW built on your system. 

OS X
----

Boost : Download Boost 1.55.0 from http://www.boost.org/ for
unix platform. Go to the download directory and uncompress it. TomoPy
won't use all the libraries so on console just run: 

./bootstrap.sh --with-libraries=system,thread,date_time

and

./b2

to build the required libraries. The build libraries are put 
inside the stage/lib folder in your boost directory.

HDF5 :

FFTW :


Linux
-----

Boost : Download Boost 1.55.0 from http://www.boost.org/ for
unix platform. Go to the download directory and uncompress it. TomoPy
won't use all the libraries so on console just run: 

./bootstrap.sh --with-libraries=system,thread,date_time

and

./b2

to build the required libraries. The build libraries are put 
inside the stage/lib folder in your boost directory.

HDF5 :

FFTW :

Windows 
-------
Boost :

HDF5 :

FFTW :
