#!/bin/bash
#SBATCH -A BIF151
#SBATCH -J score_MRI
#SBATCH -o slurm/%j.out
#SBATCH -e slurm/%j.err
#SBATCH -N 500
#SBATCH -t 04:00:00
#SBATCH -S 0
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
##SBATCH -q debug



 
module purge


module load PrgEnv-gnu
module load gcc/11.2.0
module load rocm/6.0.0
#module load craype-accel-amd-gfx90a
module load ninja

module load miniforge3

export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=https://proxy.ccs.ornl.gov:3128/
# export OMP_NUM_THREADS=2

export TORCH_HOME=$PWD/cache

source /autofs/nccs-svm1_sw/frontier/python/3.10/miniforge3/23.11.0/etc/profile.d/conda.sh

conda activate /ccs/home/palashmr/packages/miniconda/pyt_env/py310
#conda activate /ccs/home/palashmr/packages/miniconda/pyt_env/score-mri-amd

# Print Python executable and version
# which python
# python -V

#conda env list



# List installed packages to check if TensorFlow is present
#python -m pip list
export PATH=/ccs/home/palashmr/packages/miniconda/pyt_env/py310/bin:$PATH
#export PATH=/ccs/home/palashmr/packages/miniconda/pyt_env/score-mri-amd/bin:$PATH


# Test TensorFlow import and print its version
# python -c "import tensorflow as tf; print(tf.__version__)"


export LD_PRELOAD="/usr/lib64/libcrypto.so /usr/lib64/libssh.so.4 /usr/lib64/libssl.so.1.1"
# module load PrgEnv-gnu
# module load gcc/11.2.0
# module load rocm/6.0.0


export ROCM_HOME=/opt/rocm-6.0.0
#export PATH=/opt/rocm-6.0.0/bin
#export ROCM_HOME=/opt/rocm-5.6.0

#export NCCL_DEBUG=INFO
export FI_CXI_ATS=0
#export LD_LIBRARY_PATH=/opt/rocm-5.6.0/rccl/build:$PWD/aws-ofi-rccl/src/.libs/:/opt/cray/libfabric/1.15.2.0/lib64/:/opt/rocm-5.6.0/lib:/opt/rocm-5.6.0/hip/lib
export LD_LIBRARY_PATH=/opt/rocm-6.0.0/include/rccl/build:$PWD/aws-ofi-rccl/src/.libs/:/opt/cray/libfabric/1.15.2.0/lib64/:/opt/rocm-6.0.0/lib
export FI_LOG_LEVEL=info
export NCCL_NET_GDR_LEVEL=3

#export PATH="/ccs/home/palashmr/.local/crusher/miniforge3/23.11.0/bin:$PATH"



#

#required to solve MIOPEN error: https://github.com/pytorch/pytorch/issues/60477
#export MIOPEN_USER_DB_PATH="/tmp/cache"    #dont save in lustre... it will titmeout to read: https://github.com/pytorch/pytorch/issues/60477 
export MIOPEN_USER_DB_PATH="/tmp/cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

#export CUDA_VISIBLE_DEVICES=0,1

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5  #PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:128 HSA_OVERRIDE_GFX_VERSION=10.3.0 

# export MASTER_ADDR=127.0.0.1
# export MASTER_PORT=29500

export MASTER_ADDR=`ip -f inet addr show hsn0 | sed -En -e 's/.*inet ([0-9.]+).*/\1/p' | head -1`
echo "MASTER_ADDR"=$MASTER_ADDR
export NCCL_SOCKET_IFNAME=hsn
export MASTER_PORT=29500

export TORCH_EXTENSIONS_DIR="/lustre/orion/stf218/proj-shared/brave/score-MRI/temp/"  #to have a write directory
export PYTHONUNBUFFERED=1   #for immeideate printing.

export PYTORCH_ROCM_ARCH=gfx90a

#srun bash -c "env"
srun python main_fastmri.py \
 --config=/lustre/orion/stf218/proj-shared/brave/score-MRI/configs/ve/fastmri_knee_320_ncsnpp_continuous.py \
 --eval_folder=/lustre/orion/stf218/proj-shared/brave/score-MRI/workdir \
 --mode='train' \
 --workdir=/lustre/orion/stf218/proj-shared/brave/score-MRI/workdir



# srun rocgdb -x rocgdb_test.txt python

# Diagnostics to check environment settings
# echo "PYTHONPATH: $PYTHONPATH"
# echo "PATH: $PATH"
# echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

