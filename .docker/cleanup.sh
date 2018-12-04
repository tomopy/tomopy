#!/bin/bash

set -o errexit

apt-get -y autoclean
rm -rf /var/lib/apt/lists/*
rm -rf /root/*
/opt/conda/bin/conda clean -a -y
