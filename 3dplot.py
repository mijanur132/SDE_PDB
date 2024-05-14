import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data
data_3d= np.load("/raid/xrv/singlecoil_train/esc_knee_320_train/file1000443_esc.npy")
data_min = np.min(data_3d)
data_max = np.max(data_3d)
normalized_data = (data_3d - data_min) / (data_max - data_min)  # Normalization to 0-1
scaled_data = (normalized_data * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each point in the 3D space
x, y, z = data_3d.nonzero()
sc = ax.scatter(x, y, z, c=data_3d[x, y, z], cmap='viridis')

# Set axis labels
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')

# Dynamically determine range and set axis limits
max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0

mid_x = (x.max()+x.min()) * 0.5
mid_y = (y.max()+y.min()) * 0.5
mid_z = (z.max()+z.min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Optionally, to make the axes 2x larger than the data range itself:
ax.set_xlim(x.min() - max_range, x.max() + max_range)
ax.set_ylim(y.min() - max_range, y.max() + max_range)
ax.set_zlim(z.min() - max_range, z.max() + max_range)

# Colorbar (if you want to show the scale of the values)
plt.colorbar(sc)

# Save the plot
plt.savefig("3d.png")
plt.show()
