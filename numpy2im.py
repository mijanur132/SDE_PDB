from PIL import Image
import numpy as np

#load numpy array
arr=np.load("/lustre/orion/stf218/proj-shared/brave/score-MRI/results/pdb_imag_4m_real/recon/output_abs.npy")
arr=np.squeeze(arr)
print(arr,arr.shape)
arr_n=(arr-np.min(arr))/(np.max(arr)-np.min(arr))
print(arr_n)
img=Image.fromarray(np.uint8(arr_n*255)).save("/lustre/orion/stf218/proj-shared/brave/score-MRI/results/pdb_imag_4m_real/recon/output_abs.png")
