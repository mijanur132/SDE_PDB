import numpy as np
from skimage import io
from skimage.metrics import structural_similarity as ssim
from scipy.signal import fftconvolve as fc
import matplotlib.pyplot as plt
from scitbx.array_family import flex
import torch

array_in = np.load('/lustre/orion/stf218/proj-shared/brave/score-MRI/results/dnp/choice1/label/1C57_honly.mtz_0_5_sym_5.npy')
array_out = np.load('/lustre/orion/stf218/proj-shared/brave/score-MRI/results/dnp/choice1/recon/1C57_honly.mtz_0_5_1.npy')
array_sym = np.load("/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/processed/1C57_honly.mtz_0_sym_5.npy")

#total_phase = np.load ('/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/processed/1C57_phase_factor_total.npy')
total_phase_ksp = np.load('/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/processed/1C57_phase_factor.mtz_0_sym_padded_ksp_total.npy')
total_ref_unsym = np.load("/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/processed_original/1c57_honly.mtz_0_total.npy")
#total_ref = np.load('/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/processed/1C57_honly.mtz_0_sym_real_total.npy')
total_inf = np.load("/lustre/orion/stf218/proj-shared/brave/score-MRI/results/dnp/imag_mask_sym_80.0/combined120.npy")
total_ref = np.load('/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/processed/1C57_honly.mtz_0_sym_total.npy')
#total_ref = total_ref_unsym

centered= False
if centered:
    total_inf = total_inf*2-1
# print("ref:", total_ref)
# print("totalinf before:",total_inf)

#print("mean before:", np.mean(total_inf), np.std(total_inf), np.mean(total_ref), np.std(total_ref))

total_inf = np.fft.ifftn(np.fft.fft2(total_inf)) #both bef and af it has good imag value, but total_ref is real only
# print("totalinf:",total_inf)
if np.allclose(total_ref.real, total_inf.real):
    raise ValueError 
#print("fft total inf:", np.fft.fftn(total_inf))

#next two lines: remove after next inference run
# total_ref = total_ref/total_ref.std()
# total_ref_unsym = total_ref_unsym/total_ref_unsym.std()

n = 10

plt.imsave(f'/lustre/orion/stf218/proj-shared/brave/score-MRI/total_inf.png',total_inf[n].real/total_inf[n].real.std(), cmap='gray')
plt.imsave(f'/lustre/orion/stf218/proj-shared/brave/score-MRI/total_ref.png',total_ref[n].real/total_ref[n].real.std(), cmap='gray')
#before total_ref only real, total inf complex
r1=np.real(total_ref)
r2=np.real(total_inf)

x = flex.double(r1.flatten())
y = flex.double(r2.flatten())

#print("means:", np.mean(x), np.mean(y), np.std(x), np.std(y))

overall_cc = flex.linear_correlation(x = x,y = y).coefficient()
print("overall_cc:", overall_cc)

data_range=r1.max()-r1.min()
# Compute SSIM between the two arrays
ssim_index = ssim(r1, r2,data_range=data_range)
#print("SSIM_r1:", ssim_index)


#######HARDCODED slice 5###########

total_inf_ff = np.fft.ifftn(np.fft.fftn(total_inf)*total_phase_ksp)
total_ref_ff = np.fft.ifftn(np.fft.fftn(total_ref)*total_phase_ksp)
plt.imsave("/lustre/orion/stf218/proj-shared/brave/score-MRI/total_inf_ff.png",total_inf_ff.real[n]/(total_inf_ff.real[n].std()),cmap= 'gray')
print()
plt.imsave("/lustre/orion/stf218/proj-shared/brave/score-MRI/total_reff_ff.png",total_ref_ff.real[n]/total_ref_ff.real[n].std(),cmap= 'gray')
r1=total_ref_ff.real
r2=total_inf_ff.real
x = flex.double(r1.flatten())
y = flex.double(r2.flatten())

overall_cc = flex.linear_correlation(x = x,y = y).coefficient()
print("overall_cc2:", overall_cc)

total_inf_ff = total_inf_ff.real
total_ref_ff = total_ref_ff.real
data_range=total_inf_ff.max()-total_inf_ff.min()
ssim_index = ssim(total_inf_ff, total_ref_ff,data_range=data_range)
print("SSIM_r2:", ssim_index)

#print("total_ref_unsym:", total_ref_unsym)
total_ref_unsym = total_ref_unsym.real
ssim_index = ssim(total_inf_ff, total_ref_unsym, data_range=data_range)
print("SSIM_r3:", ssim_index)


#undo after we unpad before comparing IMPORTANTtttt
filter = np.abs(total_ref_unsym)>0.01*np.mean(np.abs(total_ref_unsym))
delta_phi = np.angle(np.fft.fft(total_ref_unsym[filter]))- np.angle(np.fft.fft(total_inf_ff[filter]))
C_phase=np.abs(np.mean(np.exp(1j*delta_phi)))
print("Phase correlation", C_phase)




#print(array_phase.shape, array_out.shape, array_in.shape, array_phase_ksp.shape)

# array_out_conv = fc(array_phase.real, array_out.real)
# array_in_conv = fc(array_in.real, array_phase.real)
# data_range = array_in_conv.max()-array_in_conv.min()
# ssim_index = ssim(array_in_conv, array_out_conv,data_range=data_range)
# print("SSIM_r3:", ssim_index)

# plt.imsave('/lustre/orion/stf218/proj-shared/brave/score-MRI/results/dnp/choice1/1C57_array_out_conv_.png',np.abs(array_out_conv)/array_out_conv.std(), cmap='gray')
# np.save('/lustre/orion/stf218/proj-shared/brave/score-MRI/results/dnp/choice1/1C57_array_out_conv.npy', array_out_conv)
# plt.imsave('/lustre/orion/stf218/proj-shared/brave/score-MRI/results/dnp/choice1/1C57_array_in_conv_.png',np.abs(array_in_conv)/array_in_conv.std(), cmap='gray')
# plt.imsave('/lustre/orion/stf218/proj-shared/brave/score-MRI/results/dnp/choice1/1C57_array_out_ff.png',np.abs(array_out_ff)/array_out_ff.std(), cmap='gray')
# plt.imsave('/lustre/orion/stf218/proj-shared/brave/score-MRI/results/dnp/choice1/1C57_array_in_ff.png',np.abs(array_in_ff)/array_in_ff.std(), cmap='gray')
# plt.imsave('/lustre/orion/stf218/proj-shared/brave/score-MRI/results/dnp/choice1/1C57_array_sym_ff.png',np.abs(array_sym_ff)/array_sym_ff.std(), cmap='gray')


# ssim_index = ssim(c1, c2, data_range=data_range)
# print("SSIM_c1:", ssim_index)

# ssim_index = ssim(ab1, ab2, data_range=db)
# print("SSIM_abs:", ssim_index)


# from skimage.metrics import structural_similarity as ssim
# import imageio
# from skimage.transform import resize

# # Load two images
# for i in range(1):
#     img1 = imageio.imread(f'/lustre/orion/stf218/proj-shared/brave/score-MRI/results/dnp/choice2/label/1C57_plus.mtz_0_5_sym_5.png')[:,:,0:1]
#     #img1 = imageio.imread('/lustre/orion/stf218/proj-shared/brave/score-MRI/results/n-3pol/input/1XQN_0.mtz_0_1_100.png')[:,:,0:1]
#     img2 = imageio.imread(f'/lustre/orion/stf218/proj-shared/brave/score-MRI/results/dnp/choice2/recon_progress/1C57_plus.mtz_0_5_1_3600.png')[:,:,0:1]
#     #img1 = imageio.imread('/lustre/orion/stf218/proj-shared/brave/score-MRI/results/n-3pol/recon/1XQN_0.mtz_0_1_15.png')[:,:,0:1]
#     #print(img1.shape, img2.shape)
#     img1 = img1.astype(float) / (img1.max()-img1.min())
#     img2 = img2.astype(float) / (img2.max()-img2.min())
#     # Ensure the images are the same size
#     img1 = resize(img1, (img2.shape[0], img2.shape[1]))

#     # Convert images to grayscale as SSIM is typically computed on single channel
#     data_range = img1.max() - img1.min()
#     try:
#         # Compute SSIM over all channels
#         ssim_index, ssim_map = ssim(img1, img2, multichannel=True, win_size=7, channel_axis=-1, data_range=data_range, full=True)
#         print("SSIM index:",i, ssim_index)
#     except Exception as e:
#         print("Error computing SSIM:", e)


# #label-- loss_cleared --loss2noisy_sample
# #2--1--0.98
# #4--1--0.81
# #6--0.99--0.67
# #8--0.98--0.57
# #10-- 0.97--0.48
# #15 -- 0.92--0.38
# # 20 -- 0.89--0.34
# # 25-->0.85--0.25
# #30 -- 0.83--0.28
# #35--0.81--0.26
# #40 -- 0.79--0.23
# #50 --- 0.71--0.16
# #60--0.48--0.15
# #70--0.23--0.13
# #80---0.21--0.12
# #90---0.21--0.11
# #100--0.21--0.1