import numpy as np
from PIL import Image as im
aaa=np.load("/lustre/orion/stf218/proj-shared/brave/score-MRI/workdir/samples/evaluate/sample.np")
aaa=aaa/np.std(aaa)
aaa_n=np.minimum(aaa,0)
aaa_p=np.maximum(aaa,0)
r=aaa_n[0,:,:,0]
g=aaa_p[0,:,:,0]
b=np.zeros_like(r)

r255=r*255
g255=g*255
rgb=np.stack([r255,g255,b], axis=2)
img=im.fromarray(np.uint8(rgb))
img.save("/lustre/orion/stf218/proj-shared/brave/score-MRI/workdir/samples/evaluate/rgb.png")