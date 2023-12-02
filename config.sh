#!/bin/bash


CONDA_ENV='craig_finder'

find_in_conda_env(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}

# cria o ambinente conda
if ! find_in_conda_env $CONDA_ENV ; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda env create -f environment.yml
    conda init;
    conda activate $CONDA_ENV
fi

# clona o repositório do DeepCore
if [ ! -d 'DeepCore' ]
then
    git clone https://github.com/PatrickZH/DeepCore.git
fi