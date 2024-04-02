#!/bin/bash
#SBATCH -A STF218
#SBATCH -J scoremri

#SBATCH -t 0:120:00
#SBATCH -p batch
#SBATCH -N 1

#SBATCH -o logs/scoremri-1N.o%J
#SBATCH -e logs/scoremri-1srunN.e%J


export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=https://proxy.ccs.ornl.gov:3128/
export OMP_NUM_THREADS=2

export TORCH_HOME=$PWD/cache
source /lustre/orion/world-shared/stf218/sajal/miniconda3/bin/activate
conda /lustre/orion/stf218/proj-shared/brave/score-MRI/score-mri


export LD_PRELOAD="/usr/lib64/libcrypto.so /usr/lib64/libssh.so.4 /usr/lib64/libssl.so.1.1"
module load PrgEnv-gnu
module load gcc/11.2.0
module load rocm/5.6.0


export ROCM_HOME=/opt/rocm-5.6.0

#export NCCL_DEBUG=INFO
export FI_CXI_ATS=0
export LD_LIBRARY_PATH=/opt/rocm-5.6.0/rccl/build:$PWD/aws-ofi-rccl/src/.libs/:/opt/cray/libfabric/1.15.2.0/lib64/:/opt/rocm-5.6.0/lib
export FI_LOG_LEVEL=info
export NCCL_NET_GDR_LEVEL=3

#required to solve MIOPEN error: https://github.com/pytorch/pytorch/issues/60477
#export MIOPEN_USER_DB_PATH="/tmp/cache"    #dont save in lustre... it will titmeout to read: https://github.com/pytorch/pytorch/issues/60477 
export MIOPEN_USER_DB_PATH="/tmp/cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

#DDP parameters printing
scontrol show hostnames $SLURM_NODELIST > job.node.list
input="./job.node.list"
readarray -t arr <"$input"
first=${arr[0]}
echo "first=" $first
ips=`ssh $first hostname -I`
read -ra arr <<< ${ips}
export MASTER_ADDR=${arr[0]}
echo "MASTER_ADDR=" $MASTER_ADDR

node_count=$SLURM_JOB_NUM_NODES

ranks_per_node=8
gpus_per_rank=$((8/$ranks_per_node))
ranks_total=$(($ranks_per_node*$SLURM_JOB_NUM_NODES))

srun -N$node_count -n$ranks_total -c4 python main_fastmri.py  \
 --config=/lustre/orion/stf218/proj-shared/brave/score-MRI/configs/ve/fastmri_knee_320_ncsnpp_continuous.py \
 --eval_folder=/lustre/orion/stf218/proj-shared/brave/brave_database/fastMRI/fastmri_multicoil_knee_320 \
 --mode='train'  \
 --workdir=/lustre/orion/stf218/proj-shared/brave/brave_database/fastMRI/fastmri_multicoil_knee_320




