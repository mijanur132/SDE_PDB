# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Training and evaluation for score-based generative models. """
import gc
import io
import os
import time
import glob

import numpy as np

#import tensorflow_gan as tfgan
import logging
# Keep the import below for registering all model definitions

from models import ncsnpp
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
#import evaluation
#import likelihood
import sde_lib

from absl import flags

import torch

from torch import nn

#from torch.utils import tensorboard

from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint, restore_checkpoint_disto_2_no_dist, get_mask, kspace_to_nchw, root_sum_of_squares
from utils import restore_checkpoint, get_mask, kspace_to_nchw, root_sum_of_squares

import torch.distributed as dist 
from torch.utils.data import DataLoader as DL
from torch.utils.data import DistributedSampler as DS
from torch.nn.parallel import DistributedDataParallel as DDP
#import torch.multiprocessing as mp
import argparse
import wandb 


FLAGS = flags.FLAGS
logger = logging.getLogger() 



# def init_distributed(rank,local_rank,ws,address,port):
#   dist.init_process_group(backend="nccl", init_method=f"tcp://{address}:{port}", rank=rank, world_size=ws)

#   torch.cuda.set_device(local_rank)
#   print("***************rank and world size*****************:",dist.get_rank(), dist.get_world_size()) ### most like wrong


#   #************************************************************

def train( local_rank, rank, world_size, address, port, config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  #tf.io.gfile.makedirs(sample_dir)

  #tb_dir = os.path.join(workdir, "tensorboard")
  #tf.io.gfile.makedirs(tb_dir)
  #writer = tensorboard.SummaryWriter(tb_dir)

 # init_distributed(rank,local_rank,world_size, address, port)
  device= torch.device('cuda')
  print(f"Process {rank} using device: {device}")

# Check if GPU is available
 # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  if device.type == 'cuda':
      print("GPU is available")
  else:
      print("GPU is not available, using CPU")

 
 
  # Initialize model.

  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)


  score_model = mutils.create_model(config).to(device)
  #score_model=DDP(score_model, device_ids=[local_rank])

  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0, epoch=0)


  checkpoint="/lustre/orion/stf218/proj-shared/brave/score-MRI/workdir/checkpoints/checkpoint_non_ddp.pth"
  #state=restore_checkpoint_disto_2_no_dist("/home/xrv/score-mri-palash/workdir/checkpoint_75.pth", state, config.device)
  state_dict= torch.load(checkpoint, map_location=device)
  state['model'].load_state_dict(state_dict, strict=True)
  # if skip_sigma:
  #   checkpt['model'].state_dict().pop('module.sigmas')

  

  # Build pytorch dataloader for training
  #train_loader, eval_loader= datasets.create_dataloader_ddp(config,0,world_size)

  # num_data = len(train_loader.dataset)
  # print("num data:",num_data)

  #print(train_dl.dataset.data_list), gives the names of all numpy arrays in the train data folder

  # Create data normalizer and its inverse
  # scaler = datasets.get_data_scaler(config)
  # inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")


  sampling_shape = (config.training.batch_size, config.data.num_channels,
                    config.data.image_size, config.data.image_size)
  sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)



 
  ema.store(score_model.parameters())
  ema.copy_to(score_model.parameters())
  print("sampling started..........")
  sample, n = sampling_fn(score_model)
  if config.data.is_complex:
    sample = root_sum_of_squares(sample, dim=1).unsqueeze(dim=0)
  ema.restore(score_model.parameters())
  this_sample_dir = os.path.join(sample_dir, "evaluate")
  #tf.io.gfile.makedirs(this_sample_dir)
  nrow = int(np.sqrt(sample.shape[0]))
  image_grid = make_grid(sample, nrow, padding=2)
  sample = sample.permute(0, 2, 3, 1).cpu().numpy() 
  os.makedirs(this_sample_dir, exist_ok=True)
  with open(os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
    np.save(fout, sample)

  with open(os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
    save_image(image_grid, fout)   

  print("inference ended..................")         


