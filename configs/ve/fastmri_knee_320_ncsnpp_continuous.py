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

# Lint as: python3
"""Training NCSN++ on fastmri knee with VE SDE."""

from configs.default_lsun_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'vesde'
  training.continuous = True

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'langevin'

  # data

  data = config.data
  data.khadiza=True
  data.dataset = 'fastmri_knee'
  #data.root = '/lustre/orion/stf218/proj-shared/brave/brave_database/fastMRI/singlecoil_train'
  #data.val_root='/lustre/orion/stf218/proj-shared/brave/brave_database/fastMRI/singlecoil_train'
  #data.root='/lustre/orion/stf218/proj-shared/brave/brave_database/pdb/10-20-std'
  #data.val_root='/lustre/orion/stf218/proj-shared/brave/brave_database/pdb/10-20-std-val'
  data.root = "/lustre/orion/bif151/world-shared/palashmr/n_3pol_processed"
  data.val_root = "/lustre/orion/bif151/world-shared/palashmr/n_3pol_processed_val"
  data.image_size = 320
  data.is_multi = False
  data.is_complex = False
  data.magpha=False
  #data.num_channels=2

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 4
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'residual'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3

  return config
