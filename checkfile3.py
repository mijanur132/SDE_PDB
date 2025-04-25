import numpy as np

h = np.load("/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/processed/1C57_honly.mtz_0_sym_kspace_5.npy")
m = np.load("/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/processed/1C57_minus.mtz_0_sym_kspace_5.npy")
p = np.load("/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/processed/1C57_plus.mtz_0_sym_kspace_5.npy")
s = (slice(5, 6, None), slice(5, 6, None), slice(1, 10))#10,320,320, 5,100,100
print(np.allclose(h,m))
print(np.allclose(h,p))
print(h[s])
print(m[s])
print(p[s])
print("printing abs:")
print(np.abs(h[s]))
print(np.abs(m[s]))
print(np.abs(p[s]))
print("printing angles:")
print(np.angle(h[s]))
print(np.angle(m[s]))
print(np.angle(p[s]))
