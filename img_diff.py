import imageio
import numpy as np
import matplotlib.pyplot as plt



# Load the two images as grayscale for simplicity.
# (If your images are in color, either convert them to grayscale or process each channel separately.)

for i in range(10):

    # img_real = imageio.imread(f'/lustre/orion/stf218/proj-shared/brave/score-MRI/results/protonated2/label/1c57_honly.mtz_0_5_{i}.png')[:,:,0:1]
    # img_noise = imageio.imread(f'/lustre/orion/stf218/proj-shared/brave/score-MRI/results/protonated2/label/1c57_dnp_bb.mtz_0_5_{i}.png')[:,:,0:1]
    img_real=np.load(f"/lustre/orion/stf218/proj-shared/brave/score-MRI/results/protonated2/input/1c57_honly.mtz_0_5_{i}.npy")
    img_noise=np.load(f"/lustre/orion/stf218/proj-shared/brave/score-MRI/results/protonated2/input/1c57_dnp_bb.mtz_0_5_{i}.npy")
    # Compute the pixelwise absolute difference.
    img_noise=img_noise.real
    img_noise= img_noise/img_noise.std()
    img_real=img_real.real
    img_real = img_real/img_real.std()
    diff_map = img_noise-img_real
    print(img_real.mean(),img_noise.mean(), diff_map.mean())

    # Display the difference map.
    plt.figure(figsize=(8, 6))
    plt.imshow(diff_map)
    plt.title('Pixelwise Difference Map (Noise)')
    plt.colorbar(label='Intensity Difference')
    plt.savefig(f"/lustre/orion/stf218/proj-shared/brave/score-MRI/results/protonated2/noise_plot_bb_npy_{i}.png")

