#!/bin/bash
#SBATCH -A STF218
#SBATCH -J inference_scoremri
#SBATCH -o logs/inference_%j.out  # %A: job array ID, %a: task ID
#SBATCH -e logs/inference_%j.err
#SBATCH -N 15
#SBATCH -p batch
#SBATCH -t 00:49:00
#SBATCH --cpus-per-task=4                     
##SBATCH --array=0-30 
#SBATCH --tasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH -q debug


module load PrgEnv-gnu
module load gcc/11.2.0
module load rocm/6.0.0
module load ninja
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/

source /autofs/nccs-svm1_sw/frontier/python/3.10/miniforge3/23.11.0/etc/profile.d/conda.sh

conda activate  /lustre/orion/stf218/world-shared/palashmr/py310_frontier
#export LD_LIBRARY_PATH=/opt/rocm-6.0.0/include/rccl/build:$PWD/aws-ofi-rccl/src/.libs/:/opt/cray/libfabric/1.15.2.0/lib64/:/opt/rocm-6.0.0/lib #slingshot
#export LD_LIBRARY_PATH=$PWD/aws-ofi-rccl/src/.libs/:/opt/cray/libfabric/1.15.2.0/lib64/ #slingshot, this line should be enough instead of above: need verifiction

export NCCL_NET_GDR_LEVEL=3

export MIOPEN_USER_DB_PATH="/tmp/cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

export MASTER_ADDR=`ip -f inet addr show hsn0 | sed -En -e 's/.*inet ([0-9.]+).*/\1/p' | head -1`
echo "MASTER_ADDR"=$MASTER_ADDR
export NCCL_SOCKET_IFNAME=hsn
export MASTER_PORT=29500


export TORCH_EXTENSIONS_DIR="/tmp"  #to have a write directory
export PYTHONUNBUFFERED=1   #for immeideate printing.
export PYTORCH_ROCM_ARCH=gfx90a
export WANDB_PROJECT="score-mri"

N_node=$SLURM_NNODES

# python inference_single-coil.py --data 1XQN_0.mtz_0_1 --mask_type 'gaussian2d' --acc_factor  2 --N 1500
# srun python inference_single-coil.py --data 1IDU.pdb_1_complex_10 --mask_type 'gaussian2d' --acc_factor  50

#python inference_single-coil.py --slice_idx ${SLURM_ARRAY_TASK_ID}
srun bash -c "python inference_single-coil.py --acc_factor  4 --slice_idx \${SLURM_PROCID}"
  
