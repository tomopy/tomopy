#!/bin/bash

set -o errexit

cat << EOF > /etc/profile.d/conda-libs.sh
#!/bin/bash

unset PYTHONPATH
PATH=/opt/conda/bin:${PATH}
PATH=/usr/local/cuda-10.0/NsightCompute-1.0:${PATH} 
export PATH

EOF

mkdir -p /etc/bashrc.d
cat << EOF > /etc/profile.d/conda-tomopy.sh
#!/bin/bash

if [ -z "$(which conda)" ]; then PATH="/opt/conda/bin:${PATH}"; fi
export PATH

conda init bash
# conda deactivate
conda activate tomopy

EOF
