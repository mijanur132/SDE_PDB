import numpy as np

slices =[]
for i in range(120):
    file_name=f"/lustre/orion/stf218/proj-shared/brave/score-MRI/results/dnp/vanila_4/recon/1C57_honly.mtz_0_total_{i}.npy"
    data = np.load(file_name)
    centered= True
    if centered:
        data = data*2-1

    slices.append(data)
a = np.stack(slices)
save_path=f'/lustre/orion/stf218/proj-shared/brave/score-MRI/results/dnp/vanila_4/combined120.npy'
np.save(save_path, a)




