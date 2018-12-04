#!/bin/bash

set -o errexit

cat << EOF > /etc/profile.d/conda-libs.sh
#!/bin/bash

unset PYTHONPATH
PATH=/opt/conda/bin:${PATH}
export PATH

EOF
