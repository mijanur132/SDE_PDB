import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy.signal import fftconvolve as fc
from scitbx.array_family import flex
import torch


total_inf = np.load(f"/lustre/orion/stf218/proj-shared/brave/score-MRI/results/dnp/imag_mask_uniform_sym_0.05/combined120.npy")

# total_inf = np.fft.ifftn(np.fft.fft2(total_inf)) #both bef and af it has good imag value, but total_ref is real only

total_ref = np.load('/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/processed/1C57_honly.mtz_0_sym_total.npy')#only real valued

total_ref_ksp = np.fft.fftn(total_ref)
total_inf_ksp = np.fft.fft2(total_inf)

ref_ksp_centered = np.fft.fftshift(total_ref_ksp, axes=(0,1,2))
inf_ksp_centered = np.fft.fftshift(total_inf_ksp, axes=(0,1,2))

#ref_ksp_centered_loaded_unshifted = np.load("/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/processed/1C57_honly.mtz_0_sym_padded_ksp_total.npy")
#above matches with total_ref_ksp in allclose condition

plt.imsave("/lustre/orion/stf218/proj-shared/brave/score-MRI/kspace_ref_slice_41.png",ref_ksp_centered[41].real/1, cmap='gray')
plt.imsave("/lustre/orion/stf218/proj-shared/brave/score-MRI/kspace_inf_slice_41.png",inf_ksp_centered[41].real/1, cmap='gray')

valid_slices = (
    slice(1, 120),    # axis 0
    slice(102, 219),  # axis 1
    slice(118, 203)   # axis 2
)

print("refkspcentered40:",np.sum(ref_ksp_centered[41]))
ref_valid = ref_ksp_centered[valid_slices]
inf_valid = inf_ksp_centered[valid_slices]
print("refkspcentered40afterslicing41:", np.sum(ref_valid[40]))

real = 0
if real:
    slice1 = inf_valid.real
    slice2 = ref_valid.real
else:
    slice1 = inf_valid.imag
    slice2 = ref_valid.imag


# slice1 = total_inf_ksp.real
# slice2 = total_ref_ksp.real
i = 40                               # 0‑based slice index (0 … 119)
                                      #   axis‑2 here → axial slic


# v1 = slice1[np.abs(slice1) > 1e4].ravel()      
# v2 = slice2[np.abs(slice2) > 1e-1].ravel()


# v1 = slice1[np.abs(slice1) > 1e1].ravel()      
# v2 = slice2[np.abs(slice2) > 1e-1].ravel()
v1,v2 = slice1, slice2

v1 = v1#/v1.std()
v2 = v2#/v2.std() 

print("len v1, v2:", len(v1), len(v2))

x = flex.double(v1.flatten())
y = flex.double(v2.flatten())

print("len x,y:", len(x), len(y))
#print("means:", np.mean(x), np.mean(y), np.std(x), np.std(y))

overall_cc = flex.linear_correlation(x = x,y = y).coefficient()
print("overall_cc:", overall_cc)



# v1 = ref_valid[40]
# v2 = ref_ksp_centered[41][102:219, 118:203]



n_bins = 160
# edges  = np.linspace(min(v1.min(), v2.min()),
#                      max(v1.max(), v2.max()),
#                      n_bins + 1)
edges = np.linspace(-10,10, n_bins+1)
plt.clf()
plt.figure()
plt.hist(v1.flatten(), bins=edges, density=True, alpha=0.5, label=f"inf")
plt.hist(v2.flatten(), bins=edges, density=True, alpha=0.5, label=f"ref")
#plt.xlim(-5,5)
#plt.ylim(0,1)
plt.xlabel("Scattering lenght (h density)");  plt.ylabel("Probability density")
plt.title(f"Histogram comparison")
plt.legend(); plt.tight_layout()
plt.savefig("/lustre/orion/stf218/proj-shared/brave/score-MRI/hist_real_slice.png")