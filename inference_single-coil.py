from pathlib import Path
from models import utils as mutils
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector,
                      get_pc_fouriercs_RI)
from models import ncsnpp
import time
from utils import fft2, ifft2, get_mask, get_data_scaler, get_data_inverse_scaler, restore_checkpoint
import torch
import torch.nn as nn
import numpy as np
from models.ema import ExponentialMovingAverage
import matplotlib.pyplot as plt
import importlib
import argparse


def main():
    ###############################################
    # 1. Configurations
    ###############################################

    # args
    args = create_argparser().parse_args()
    N = args.N
    m = args.m
    fname = args.data
    #filename = f'./samples/single-coil/{fname}.npy'
    filename = f'./samples/n-3pol/{fname}.npy'
    #filename = f'./samples/mesolite/{fname}.npy'   #real space (real valued)
    #filename=    f"/lustre/orion/stf218/proj-shared/brave/brave_database/COD/320/validation/1001169.cif_0_1.npy"


    print('initaializing...')
    configs = importlib.import_module(f"configs.ve.fastmri_knee_320_ncsnpp_continuous")
    config = configs.get_config()
    img_size = config.data.image_size
    batch_size = 1

    # Read data
    img = torch.from_numpy(np.load(filename))[0]#.astype(np.complex64))
    print(img.shape)
    img = img.view(1, 1, 320, 320)
    img = img.to(config.device)

    mask = get_mask(img, img_size, batch_size,
                    type=args.mask_type,
                    acc_factor=args.acc_factor,
                    center_fraction=args.center_fraction)

    ckpt_filename=f"/lustre/orion/stf218/proj-shared/brave/score-MRI/workdir/checkpoints/non_ddp_checkpoint_238_249.pth"
    #ckpt_filename=f"/lustre/orion/stf218/proj-shared/brave/score-MRI/workdir/checkpoints/non_ddp_checkpoint_5_19.pth"
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=N)

    config.training.batch_size = batch_size
    predictor = ReverseDiffusionPredictor
    corrector = LangevinCorrector
    probability_flow = False
    snr = 0.16

    # sigmas = mutils.get_sigmas(config)
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # create model and load checkpoint
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(),
                                   decay=config.model.ema_rate)
    state = dict(step=0, model=score_model, ema=ema)
    
    checkpt = torch.load(ckpt_filename, map_location=config.device)
 
    state['model'].load_state_dict(checkpt['model'], strict=False)
    state['ema'].load_state_dict(checkpt['ema'])
    ema.copy_to(score_model.parameters())

    # Specify save directory for saving generated samples
    #save_root = Path(f'./results/single-coil')
    save_root = Path(f'./results/n-3pol')
    #save_root = Path(f'./results/mesolite/g2d_3rd')
    save_root.mkdir(parents=True, exist_ok=True)

    irl_types = ['input', 'recon', 'recon_progress', 'label']
    for t in irl_types:
        save_root_f = save_root / t
        save_root_f.mkdir(parents=True, exist_ok=True)

    ###############################################
    # 2. Inference
    ###############################################

    pc_fouriercs = get_pc_fouriercs_RI(sde,
                                       predictor, corrector,
                                       inverse_scaler,
                                       snr=snr,
                                       n_steps=m,
                                       probability_flow=probability_flow,
                                       continuous=config.training.continuous,
                                       denoise=True, save_root=save_root, f_name =f'{fname}_{args.acc_factor}')
    # fft
    kspace = fft2(img)   #reciprocal space

    under_kspace = kspace * mask  #multiplicative mask, 1 means present

    # r=torch.real(kspace).float()
    # under_kspace=torch.complex(r,torch.zeros_like(r)) #reciprocal, replace imaginary part with zeros

    under_img = ifft2(under_kspace) #back to real space
    
    print(f'Beginning inference')
    tic = time.time()
    x = pc_fouriercs(score_model, under_img, mask, Fy=under_kspace)
    toc = time.time() - tic
    print(f'Time took for recon: {toc} secs.')

    ###############################################
    # 3. Saving recon
    ###############################################
    input = under_img.squeeze().cpu().detach().numpy()
    label = img.squeeze().cpu().detach().numpy()
    mask_sv = mask.squeeze().cpu().detach().numpy()

    np.save(f'{save_root}/input/{fname}_{args.acc_factor}.npy', input)
    np.save(f'{save_root}/input/{fname}_{args.acc_factor}_mask.npy', mask_sv)
    np.save(str(save_root / 'label' / fname) + '.npy', label)
    plt.imsave(f'{save_root}/label/{fname}.png',np.abs(label)*30, cmap='gray')
    plt.imsave(f'{save_root}/input/{fname}_{args.acc_factor}.png', np.abs(input)*50, cmap='gray')
    plt.imsave(f'{save_root}/input/{fname}_{args.acc_factor}_mask.png', np.abs(mask_sv), cmap='gray')

    recon = x.squeeze().cpu().detach().numpy()
    np.save(f'{save_root}/recon/{fname}_{args.acc_factor}.npy', recon)
    plt.imsave(f'{save_root}/recon/{fname}_{args.acc_factor}.png', np.abs(recon)*30, cmap='gray')


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='which data to use for reconstruction', required=True)
    parser.add_argument('--mask_type', type=str, help='which mask to use for retrospective undersampling.'
                                                      '(NOTE) only used for retrospective model!', default='gaussian2d',
                        choices=['gaussian1d', 'uniform1d', 'gaussian2d'])
    parser.add_argument('--acc_factor', type=int, help='Acceleration factor for Fourier undersampling.'
                                                       '(NOTE) only used for retrospective model!', default=4)
    parser.add_argument('--center_fraction', type=float, help='Fraction of ACS region to keep.'
                                                       '(NOTE) only used for retrospective model!', default=0.08)
    parser.add_argument('--save_dir', default='./results')
    parser.add_argument('--N', type=int, help='Number of iterations for score-POCS sampling', default=1500)
    parser.add_argument('--m', type=int, help='Number of corrector step per single predictor step.'
                                              'It is advised not to change this default value.', default=1)
    return parser


if __name__ == "__main__":
    main()

#for slurm: sbatch train_frontier.slurm <acc_factor>
#acc factor 1 to inf, larger mean more noise
#For terminal: train_script.sh
#for pdb checkpoint:     ckpt_filename=f"/lustre/orion/stf218/proj-shared/brave/score-MRI/workdir/checkpoints/non_ddp_checkpoint_58_15.pth"
#mesolite   #ckpt_filename=f"/lustre/orion/stf218/proj-shared/brave/score-MRI/workdir/checkpoints/non_ddp_checkpoint_5_19.pth"
#change data location in config file