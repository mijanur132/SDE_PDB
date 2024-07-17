#!/bin/bash

# Environment and module setup
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=https://proxy.ccs.ornl.gov:3128/
export OMP_NUM_THREADS=2

export TORCH_HOME=$PWD/cache
module load miniforge3
source /autofs/nccs-svm1_sw/frontier/python/3.10/miniforge3/23.11.0/etc/profile.d/conda.sh

conda activate /ccs/home/palashmr/packages/miniconda/pyt_env/score-mri-amd

export PATH=/ccs/home/palashmr/packages/miniconda/pyt_env/score-mri-amd/bin:$PATH



python -c "import tensorflow as tf; print(tf.__version__)"

export LD_PRELOAD="/usr/lib64/libcrypto.so /usr/lib64/libssh.so.4 /usr/lib64/libssl.so.1.1"

module load PrgEnv-gnu
module load gcc/11.2.0
module load amd-mixed/6.0.0
module load craype-accel-amd-gfx90a
module load ninja

export ROCM_HOME=/opt/rocm-6.0.0
export NCCL_DEBUG=INFO
export FI_CXI_ATS=0
export LD_LIBRARY_PATH=/opt/rocm-6.0.0/include/rccl/build:$PWD/aws-ofi-rccl/src/.libs/:/opt/cray/libfabric/1.15.2.0/lib64/:/opt/rocm-6.0.0/lib
export FI_LOG_LEVEL=info
export NCCL_NET_GDR_LEVEL=3

export MIOPEN_USER_DB_PATH="/tmp/cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

# DDP setup
scontrol show hostnames $SLURM_NODELIST > job.node.list
input="./job.node.list"
readarray -t arr <"$input"
first=${arr[0]}
echo "first=" $first
ips=`ssh $first hostname -I`
read -ra arr <<< ${ips}
export MASTER_ADDR=${arr[0]}
echo "MASTER_ADDR=" $MASTER_ADDR
export MASTER_PORT=29500

node_count=$SLURM_JOB_NUM_NODES
ranks_per_node=8
gpus_per_rank=$((8/$ranks_per_node))
ranks_total=$(($ranks_per_node*$node_count))
echo "node_count=" $node_count

export CUDA_VISIBLE_DEVICES=0

# Run the Python script
srun -N$node_count -n$ranks_total -c4 /ccs/home/palashmr/packages/miniconda/pyt_env/score-mri-amd/bin/python main_fastmri.py \
 --config=/lustre/orion/stf218/proj-shared/brave/score-MRI/configs/ve/fastmri_knee_320_ncsnpp_continuous.py \
 --eval_folder=/lustre/orion/stf218/proj-shared/brave/score-MRI/workdir \
 --mode='train' \
 --workdir=/lustre/orion/stf218/proj-shared/brave/score-MRI/workdir
