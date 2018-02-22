FROM continuumio/miniconda3

WORKDIR tmp
RUN \
    conda install numpy -y \
    && conda install pandas -y \
    && conda install scipy -y \
