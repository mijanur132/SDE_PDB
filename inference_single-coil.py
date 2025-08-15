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
import sys

def rgb2bit(image, b_type, b_scale, perm=None, gray_codes=None):
    
    if images.dtype != torch.uint8:
            raise ValueError("images must be uint8 dtype")

        if b_type == 'gray':
            if gray_codes is None:
                raise ValueError("gray_codes is required for b_type='gray'")
            images = gray_codes[images]
        elif b_type != 'uint8':
            raise ValueError(f"Unsupported b_type: {b_type}")

        B, H, W, C = images.shape
        bits = ((images.unsqueeze(-1) >> torch.arange(7, -1, -1)) & 1).float()  # [B,H,W,3,8]
        bits = bits.view(B, H, W, -1)  # [B,H,W,24]
        bits = (bits * 2 - 1) * b_scale

        return bits

def bit2rgb_torch(bits, b_type, b_scale=1.0, gray_inv_codes=None):
    """
    Convert bitwise encoded tensor back to RGB image (uint8).

    Args:
        bits (Tensor): [B, H, W, 24] in [-b_scale, +b_scale]
        b_type (str): 'uint8' or 'gray'
        b_scale (float): must match what was used in rgb2bit
        gray_inv_codes (Tensor): required if b_type == 'gray', shape [256]

    Returns:
        Tensor: [B, H, W, 3], dtype uint8
    """
    if bits.shape[-1] != 24:
        raise ValueError("Expected 24 channels (8 bits per RGB channel)")

    # Unscale and threshold bits → binary
    bits = (bits / b_scale).clamp(-1, 1)
    bits = ((bits + 1) / 2 > 0.5).to(torch.uint8)  # [B, H, W, 24]

    # Reshape to 3 × 8
    bits = bits.view(*bits.shape[:-1], 3, 8)  # [B, H, W, 3, 8]

    # Convert 8 bits → int
    powers = (1 << torch.arange(7, -1, -1, device=bits.device)).to(torch.uint8)
    ints = torch.sum(bits * powers, dim=-1)  # [B, H, W, 3]

    if b_type == 'gray':
        if gray_inv_codes is None:
            raise ValueError("gray_inv_codes required for b_type='gray'")
        ints = gray_inv_codes[ints]  # reverse Gray code

    elif b_type != 'uint8':
        raise ValueError(f"Unsupported b_type: {b_type}")

    return ints  # dtype: uint8, shape [B, H, W, 3]


def main():
    ###############################################
    # 1. Configurations
    ###############################################

    # args
    args = create_argparser().parse_args()
    N = args.N
    m = args.m
    slice_idx = args.slice_idx#+60
    print("sys.argv:", sys.argv, slice_idx)

    fname ='1C57_honly.mtz_0_total' #args.data
    #filename = f'./samples/single-coil/{fname}.npy'
    #filename = f'./samples/n-3pol/{fname}.npy'
    #filename = f'./samples/protonated/{fname}.npy'
    #filename = f'./samples/mesolite/{fname}.npy'   #real space (real valued)
    #filename=    f"/lustre/orion/stf218/proj-shared/brave/brave_database/COD/320/validation/1001169.cif_0_1.npy"
    #filename_plus = f'/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/processed/1C57_plus.mtz_0_sym_5.npy'
    #filename_ref = f'/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/processed/1C57_honly.mtz_0_sym_total.npy'
    #filename_minus = f'/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/processed/1C57_minus.mtz_0_sym_5.npy'
    filename_ref = f'/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/processed/1C57_honly.mtz_0_sym_total.npy'
    #filename_ref = f'/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/processed_original/1c57_honly.mtz_0_total.npy'
    print('initaializing...')
    configs = importlib.import_module(f"configs.ve.fastmri_knee_320_ncsnpp_continuous")
    config = configs.get_config()
    img_size = config.data.image_size
    batch_size = 1

    # Read data
    option = 1
    if option==1:
        filename = filename_ref
    else:
        raise ValueError
        #filename = filename_ref #implement random alternative signage approach 2
    imgx = torch.from_numpy(np.load(filename))#.astype(np.complex64)
    imgx = torch.fft.ifft2(torch.fft.fftn(imgx))#complex
    #previoulsy ksp to 3d fftn to image
    #now we first reverse back to ksp and do 2d ifft2 to back to image, 
    i = slice_idx
    img = imgx[slice_idx]
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
    print(" args.acc_factor", args.acc_factor)
    save_root = Path(f'./results/dnp/imag_mask_sym_{args.acc_factor}')
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
    #under_kspace = kspace * mask  #multiplicative mask, 1 means present
    #print("kspace before:", kspace[ 0,   0, 129, 161]) #index gives nonzero imag for slice 50
    real = torch.real(kspace)
    imag = torch.imag(kspace)
    masked_imag = imag*mask 
    under_kspace = real + 1j*masked_imag
    #print("kspace after:", under_kspace[ 0,   0, 129, 161])
  
    under_img = ifft2(under_kspace) #back to real space
    #under_kspace = torch.real(under_kspace) #appraoch 1
    
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

    np.save(f'{save_root}/input/{fname}_sym_{i}.npy', input)
    #np.save(f'{save_root}/input/{fname}_{args.acc_factor}_mask.npy', mask_sv)
    np.save(f'{save_root}/label/{fname}_sym_{i}.npy', label)
    
    plt.imsave(f'{save_root}/label/{fname}_sym_{i}.png',np.abs(label)/label.std(), cmap='gray')
    plt.imsave(f'{save_root}/input/{fname}_sym_{i}.png', np.abs(input)/input.std(), cmap='gray')
    #plt.imsave(f'{save_root}/input/{fname}_{args.acc_factor}_mask.png', np.abs(mask_sv), cmap='gray')

    recon = x.squeeze().cpu().detach().numpy()
    np.save(f'{save_root}/recon/{fname}_{i}.npy', recon)
    plt.imsave(f'{save_root}/recon/{fname}_{i}.png', np.abs(recon)/recon.std(), cmap='gray')


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='which data to use for reconstruction', required=False)
    parser.add_argument('--mask_type', type=str, help='which mask to use for retrospective undersampling.'
                                                    '(NOTE) only used for retrospective model!', default='gaussian2d',
                        choices=['gaussian1d', 'uniform1d', 'gaussian2d'])
    parser.add_argument('--acc_factor', type=float, help='Acceleration factor for Fourier undersampling.'
                                                    '(NOTE) only used for retrospective model!', default=20)
    parser.add_argument('--center_fraction', type=float, help='Fraction of ACS region to keep.'
                                                    '(NOTE) only used for retrospective model!', default=0.08)
    parser.add_argument('--save_dir', default='./results')
    parser.add_argument('--N', type=int, help='Number of iterations for score-POCS sampling', default=4000)
    parser.add_argument('--m', type=int, help='Number of corrector step per single predictor step.'
                                            'It is advised not to change this default value.', default=1)
    parser.add_argument('--slice_idx', type = int, help= 'which slice being processed.', default= 5)
    return parser


if __name__ == "__main__":
    main()

#for slurm: sbatch train_frontier.slurm <acc_factor>
#acc factor 1 to inf, larger mean more noise
#For terminal: train_script.sh
#for pdb checkpoint:     ckpt_filename=f"/lustre/orion/stf218/proj-shared/brave/score-MRI/workdir/checkpoints/non_ddp_checkpoint_58_15.pth"
#mesolite   #ckpt_filename=f"/lustre/orion/stf218/proj-shared/brave/score-MRI/workdir/checkpoints/non_ddp_checkpoint_5_19.pth"
#change data location in config file