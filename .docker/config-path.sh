#!/bin/bash

set -o errexit

cat << EOF > /etc/profile.d/nsight-compute.sh

__cuda_path=/usr/local/cuda

if [ -z "${__cuda_path}" ]; then 
   echo "No CUDA path found @ ${__cuda_path}!"
else
    __nsight_path=${__cuda_path}/NsightCompute-1.0
    if [ -d "${__nsight_path}" ]; then
        echo -e "Adding ${__nsight_path} to PATH..."
        PATH=${__nsight_path}:${PATH} 
    else
        echo -e "${__nsight_path} not found!"
    fi
    unset __nsight_path
fi

unset __cuda_path
export PATH
 
EOF

cat << EOF > /etc/profile.d/conda-libs.sh
#!/bin/bash

unset PYTHONPATH

echo -e "Adding conda to path..."
PATH=/opt/conda/bin:${PATH}
export PATH

echo -e "Initializing conda..."
conda activate

EOF

mkdir -p /etc/bashrc.d

cat << EOF > /etc/bashrc.d/conda-tomopy.sh
#!/bin/bash

if [ -z "$(which conda)" ]; then 
   . /etc/profile.d/conda-libs.sh
fi

echo -e "Activating tomopy..."
conda activate tomopy

EOF
