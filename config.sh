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

{
    
    # try
    pip install cords

} || {

    # catch

    if [ ! -d 'cords' ]
    then
        git clone https://github.com/decile-team/cords.git
    fi
    pip install -r cords/requirements/requirements.txt
    python cords/setup.py build
    python cords/setup.py install

}