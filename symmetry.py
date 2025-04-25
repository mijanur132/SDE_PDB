import numpy as np


x_plus = np.load('/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/original/1C57_plus.mtz_0.npy')   # shape (x, y, z)
x_minus = np.load('/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/original/1C57_minus.mtz_0.npy')  # shape (x, y, z)
h_only = np.load("/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/original/1c57_honly.mtz_0.npy")
mag_x = 0.5 * (np.abs(x_plus) + np.abs(x_minus))
mag_honly = np.abs(h_only)
phi_plus = np.angle(x_plus)
phi_minus = np.angle(x_minus)
phi_honly =np.angle(h_only)
phi_0 = 0.5 * (phi_plus + phi_minus)
x_plus_new = mag_x * (np.cos(phi_plus - phi_0) + 1j * np.sin(phi_plus - phi_0))
x_minus_new = mag_x * (np.cos(phi_minus - phi_0) + 1j * np.sin(phi_minus - phi_0))
h_only_new = mag_honly * (np.cos(phi_honly - phi_0) + 1j * np.sin(phi_honly - phi_0))
factor = np.cos(phi_0)+1j*np.sin(phi_0)



np.save('/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/symmetrized/1C57_plus.mtz_0_sym.npy', x_plus_new)
np.save('/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/symmetrized/1C57_minus.mtz_0_sym.npy', x_minus_new)
np.save('/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/symmetrized/1C57_honly.mtz_0_sym.npy', h_only_new)
np.save('/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/symmetrized/1C57_phase_factor.mtz_0_sym.npy',factor)
np.save('/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/symmetrized/1C57_honly.mtz_0_sym_real.npy', h_only_new.real)
np.save('/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/symmetrized/1C57_honly.mtz_0_real.npy', h_only.real)

print("New files 'x_plus_new.npy' and 'x_minus_new.npy' have been saved successfully.")



