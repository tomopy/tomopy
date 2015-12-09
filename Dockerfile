
# http://docs.docker.com/engine/reference/builder/
# https://github.com/ContinuumIO/docker-images/tree/master/miniconda
FROM continuumio/miniconda

RUN conda install -c dgursoy tomopy