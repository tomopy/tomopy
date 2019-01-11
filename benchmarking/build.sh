#!/bin/bash -e

DIR=${PWD}
while [ ! -f setup.py ]
do
    if [ "${DIR}" = "/" ]; then exit 1; fi
    cd ..
done

python setup.py install $@ 1> /dev/null

echo -e "Build success"
