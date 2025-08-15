import numpy as np

import numpy as np
import os
import glob

# Input and output folders
input_dir = '/lustre/orion/stf218/world-shared/palashmr/npy_files_jul23_processed'
output_dir = input_dir + '_symmetrized'
os.makedirs(output_dir, exist_ok=True)

# Get all "plus" files to drive the loop
plus_files = sorted(glob.glob(os.path.join(input_dir, '*_plus.mtz_0_*.npy')))

for plus_path in plus_files:
    # Derive basename and shared key (e.g., "1C57" and "1")
    base = os.path.basename(plus_path)
   

    prefix, suffix = base.split('_plus')  # e.g., '9BPE', '.mtz_0_8.npy'
    suffix = suffix.replace('.npy', '')   # e.g., '.mtz_0_8'

    # Construct corresponding filenames
    minus_path = os.path.join(input_dir, f'{prefix}_minus{suffix}.npy')
    honly_path = os.path.join(input_dir, f'{prefix}_Honly{suffix}.npy')
    key = f'{prefix}{suffix}'  # used for output filenames

    # Skip if corresponding files not found
    if not (os.path.exists(minus_path) and os.path.exists(honly_path)):
        print(minus_path)
        print(honly_path)
        print(f"Skipping set: {key} (missing files)")
        continue

    # Load data
    x_plus = np.load(plus_path)
    x_minus = np.load(minus_path)
    h_only = np.load(honly_path)

    # Compute magnitudes and phases
    mag_x = 0.5 * (np.abs(x_plus) + np.abs(x_minus))
    mag_honly = np.abs(h_only)

    phi_plus = np.angle(x_plus)
    phi_minus = np.angle(x_minus)
    phi_honly = np.angle(h_only)
    phi_0 = 0.5 * (phi_plus + phi_minus)

    # Apply phase alignment
    x_plus_new = mag_x * (np.cos(phi_plus - phi_0) + 1j * np.sin(phi_plus - phi_0))
    x_minus_new = mag_x * (np.cos(phi_minus - phi_0) + 1j * np.sin(phi_minus - phi_0))
    h_only_new = mag_honly * (np.cos(phi_honly - phi_0) + 1j * np.sin(phi_honly - phi_0))
    factor = np.cos(phi_0) + 1j * np.sin(phi_0)

    # Save outputs
    np.save(os.path.join(output_dir, f'{key}_plus_sym.npy'), x_plus_new)
    np.save(os.path.join(output_dir, f'{key}_minus_sym.npy'), x_minus_new)
    np.save(os.path.join(output_dir, f'{key}_honly_sym.npy'), h_only_new)
    np.save(os.path.join(output_dir, f'{key}_phase_factor.npy'), factor)
    np.save(os.path.join(output_dir, f'{key}_honly_sym_real.npy'), h_only_new.real)
    np.save(os.path.join(output_dir, f'{key}_honly_real.npy'), h_only.real)

    print(f"Processed: {key}")

print("\n Batch symmetrization completed.")


# x_plus = np.load('/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/original/1C57_plus.mtz_0.npy')   # shape (x, y, z)
# # x_minus = np.load('/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/original/1C57_minus.mtz_0.npy')  # shape (x, y, z)
# # h_only = np.load("/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/original/1c57_honly.mtz_0.npy")
# # mag_x = 0.5 * (np.abs(x_plus) + np.abs(x_minus))
# # mag_honly = np.abs(h_only)
# # phi_plus = np.angle(x_plus)
# # phi_minus = np.angle(x_minus)
# # phi_honly =np.angle(h_only)
# # phi_0 = 0.5 * (phi_plus + phi_minus)
# # x_plus_new = mag_x * (np.cos(phi_plus - phi_0) + 1j * np.sin(phi_plus - phi_0))
# # x_minus_new = mag_x * (np.cos(phi_minus - phi_0) + 1j * np.sin(phi_minus - phi_0))
# # h_only_new = mag_honly * (np.cos(phi_honly - phi_0) + 1j * np.sin(phi_honly - phi_0))
# # factor = np.cos(phi_0)+1j*np.sin(phi_0)



# # np.save('/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/symmetrized/1C57_plus.mtz_0_sym.npy', x_plus_new)
# # np.save('/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/symmetrized/1C57_minus.mtz_0_sym.npy', x_minus_new)
# # np.save('/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/symmetrized/1C57_honly.mtz_0_sym.npy', h_only_new)
# # np.save('/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/symmetrized/1C57_phase_factor.mtz_0_sym.npy',factor)
# # np.save('/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/symmetrized/1C57_honly.mtz_0_sym_real.npy', h_only_new.real)
# # np.save('/lustre/orion/stf218/proj-shared/brave/score-MRI/samples/dnp/symmetrized/1C57_honly.mtz_0_real.npy', h_only.real)

# # print("New files 'x_plus_new.npy' and 'x_minus_new.npy' have been saved successfully.")



