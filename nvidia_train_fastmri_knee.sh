#!/bin/bash 



# export http_proxy=http://proxy.ccs.ornl.gov:3128/
# export https_proxy=https://proxy.ccs.ornl.gov:3128/
# export OMP_NUM_THREADS=2

export TORCH_HOME=$PWD/cache


#export LD_PRELOAD="/usr/lib64/libcrypto.so /usr/lib64/libssh.so.4 /usr/lib64/libssl.so.1.1"

module load PrgEnv-gnu
module load gcc/11.2.0
# module load rocm/6.0.0

#module load PrgEnv-gnu
#module load gcc/11.2.0





export CUDA_VISIBLE_DEVICES=6

python main_fastmri.py \
 --config=~/score-mri-palash/configs/ve/fastmri_knee_320_ncsnpp_continuous.py\
 --eval_folder=~/score-mri-palash/workdir\
 --mode='train'  \
 --workdir=~/score-mri-palash/workdir
