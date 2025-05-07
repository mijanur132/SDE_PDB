import numpy as np
acc = 0.75
slices =[]
for i in range(120):
    file_name=f"/lustre/orion/stf218/proj-shared/brave/score-MRI/results/dnp/imag_mask_sym_{acc}/recon/1C57_honly.mtz_0_total_{i}.npy"
    data = np.load(file_name)
    centered= True
    if centered:
        data = data*2-1

    slices.append(data)
a = np.stack(slices)
save_path=f'/lustre/orion/stf218/proj-shared/brave/score-MRI/results/dnp/imag_mask_sym_{acc}/combined120.npy'
np.save(save_path, a)

# for i in range(120):
#     file_name=f"/lustre/orion/stf218/proj-shared/brave/score-MRI/results/dnp/vanila_4/input/1C57_honly.mtz_0_total_sym_{i}.npy"
#     data = np.load(file_name)
#     centered= True
#     if centered:
#         data = data*2-1

#     slices.append(data)
# a = np.stack(slices)
# save_path=f'/lustre/orion/stf218/proj-shared/brave/score-MRI/results/dnp/vanila_6/input/combined120.npy'
# np.save(save_path, a)




