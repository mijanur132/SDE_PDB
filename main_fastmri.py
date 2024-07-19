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

"""Training and evaluation"""

import run_lib_fastmri
print("before app")
from absl import app
print("after app")
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os
#import tensorflow as tf
#import torch.multiprocessing as mp
# ##
# import pdb_attach
# pdb_attach.listen(50000)
# ###



# ########
# import pdb_attach
# pdb_attach.listen(50000)  # Listen on port 50000.

# ####



FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "train_regression", "eval"], "Running mode: train, train_regression, or eval")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.mark_flags_as_required(["workdir", "config", "mode"])

# def train_wrapper(rank, world_size, config, workdir):
#   run_lib_fastmri.train(rank, world_size, config, workdir)

def main(argv):
  # visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
  # if visible_devices is None:
  #     raise ValueError("No GPUs specified in CUDA_VISIBLE_DEVICES")
  
  # gpus = list(map(int, visible_devices.split(',')))
  # gpus = list(map(int, visible_devices.split(',')))
  # world_size = len(gpus)

  print(" entered main.........")

  if "SLURM_NTASKS" in os.environ:

    world_size=int(os.environ["SLURM_NTASKS"])
    rank=int(os.environ["SLURM_PROCID"])
    address=os.environ["MASTER_ADDR"]
    port=os.environ["MASTER_PORT"]

    print(f"world size and rank:{world_size}, {rank}")
  else:
    world_size=1
    rank=0
    address="127.0.0.1"
    port=29500


  print(FLAGS.config)

  if FLAGS.mode == "train" or FLAGS.mode == "train_regression":
    # Create the working directory
    #tf.io.gfile.makedirs(FLAGS.workdir)
    # Set logger so that it outputs to both console and file
    # Make logging work for both disk and Google Cloud Storage
    gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')
    # Run the training pipeline
    
    if FLAGS.mode == "train":
     
      print(f"train..ws:{world_size}, rank:{rank}")
      run_lib_fastmri.train(rank,world_size, address,port, FLAGS.config, FLAGS.workdir)
     

    elif FLAGS.mode == "train_regression":
      run_lib_fastmri.train_regression(FLAGS.config, FLAGS.workdir)
  elif FLAGS.mode == "eval":
    # Run the evaluation pipeline
    run_lib_fastmri.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
  print("before main is called.......")
  app.run(main)
