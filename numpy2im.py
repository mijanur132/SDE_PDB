from PIL import Image
import numpy as np

#load numpy array
arr=np.load("/home/xrv/score-mri-palash/workdir/samples/iter_190/sample.np")
arr=np.squeeze(arr)
print(arr,arr.shape)
arr_n=(arr-np.min(arr))/(np.max(arr)-np.min(arr))
print(arr_n)
img=Image.fromarray(np.uint8(arr_n*255)).save("/home/xrv/score-mri-palash/workdir/samples/iter_190/sample1.png")
