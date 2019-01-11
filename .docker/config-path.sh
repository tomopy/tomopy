#!/bin/bash

set -o errexit

cat << EOF > /etc/profile.d/conda-libs.sh
#!/bin/bash

unset PYTHONPATH
PATH=/opt/conda/bin:${PATH}
export PATH

EOF

mkdir -p /etc/bashrc.d
cat << EOF > /etc/profile.d/conda-tomopy.sh
#!/bin/bash

if [ -z "$(which conda)" ]; then PATH="/opt/conda/bin:${PATH}"; fi
export PATH

source deactivate
source activate tomopy

EOF
