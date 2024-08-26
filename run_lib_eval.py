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
from utils import save_checkpoint, restore_checkpoint, restore_checkpoint_disto_2_no_dist, get_mask, kspace_to_nchw, root_sum_of_squares, save_checkpoint_for_non_ddp
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


def init_distributed(rank,local_rank,ws,address,port):
  dist.init_process_group(backend="nccl", init_method=f"tcp://{address}:{port}", rank=rank, world_size=ws)
  torch.cuda.set_device(local_rank)
  print("***************rank and world size*****************:",dist.get_rank(), dist.get_world_size()) ### most like wrong


  #************************************************************

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

  init_distributed(rank,local_rank,world_size, address, port)
  device= torch.device('cuda', local_rank)
  print(f"Process {rank} using device: {device}")

  if device.type == 'cuda':
      print("GPU is available")
  else:
      print("GPU is not available, using CPU")

  # Initialize model.

  score_model = mutils.create_model(config).to(device)
  score_model=DDP(score_model, device_ids=[local_rank])

  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0, epoch=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pth"))
  checkpoint_files.sort(key=os.path.getmtime, reverse=True)
  initial_step = int(state['step'])
  initial_epoch = int(state['epoch'])

  if checkpoint_files:
    latest_checkpoint = checkpoint_files[0]
    if rank==0:
      print(f"latest checkpoint.................:{latest_checkpoint}")
    checkpoint_dir_temp = os.path.join(workdir, "checkpoints", latest_checkpoint)
    state = restore_checkpoint(checkpoint_dir_temp, state, config.device)
    initial_epoch = int(state['epoch'])+1
    initial_step = int(state['step'])+1
  else:
      latest_checkpoint = None
      print("No checkpoint files found.")

  # Build pytorch dataloader for training
  train_loader, eval_loader= datasets.create_dataloader_ddp(config,rank,world_size)
  num_data = len(train_loader.dataset)
  if rank==0:
    print(f"train data: {num_data}, validation data: {eval_loader}")

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

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

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)

  # Building sampling functions
  if config.training.snapshot_sampling:
    sampling_shape = (config.training.batch_size, config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  if rank==0:
    logging.info(f"Starting training loop at epoch: {initial_epoch},step:{initial_step}")

  loss=0
  if rank==0:
    config_dict=dict(config.items())
    config_dict["total_batch_size"]=config.training.batch_size*world_size
    wandb.init(config=config_dict)
  
  for epoch in range(initial_epoch, config.training.epochs):
    train_loader.sampler.set_epoch(epoch)
    if rank==0:
      print('=================================================')
      print(f'Epoch:............................................... {epoch}')
      print('=================================================')

    for step, batch in enumerate(train_loader, start=1):

      batch = scaler(batch.to(device))
      batch=batch.real
      batch=batch.float()
      batch=torch.transpose(batch,0,1)
      images=torch.unbind(batch,dim=0)
      images = [img.unsqueeze(0) for img in images]
      nimg=4
      # for i in range(0, len(images)-1*nimg, nimg):
      #     batch_images = images[i:i+nimg]
      #     imgs = torch.cat(batch_images, dim=0)  # 4,1,320,320
      #     loss = eval_step_fn(state, imgs)
      global_step = num_data * epoch + step
      # if step % config.training.log_freq == 0 and rank==0:
      #   logging.info("epoch: %d, step: %d, training_loss: %.5e" % (epoch,step, loss.item()))
      #   if rank==0:
      #     wandb.log({"step": global_step, "loss": loss})

    
      eval_batch = scaler(next(iter(eval_loader)).to(device))
      eval_batch=eval_batch.real
      eval_batch=eval_batch.float()
      eval_batch=torch.transpose(eval_batch,0,1)
      eval_images=torch.unbind(eval_batch,dim=0)
      eval_images = [img.unsqueeze(0) for img in eval_images]

      #eval_images = torch.unbind(torch.transpose(scaler(next(iter(eval_loader)).to(device)).real.float(), 0, 1), dim=0)
      eval_imgs = torch.cat(eval_images[0:19], dim=0) 
      eval_loss = eval_step_fn(state, eval_imgs)
      if rank==0:
          logging.info("epoch:%d, step: %d, eval_loss: %.5e" % (epoch,step, eval_loss.item()))
          wandb.log({"step": global_step, "eval_loss": eval_loss.item()})
