import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Load your loss data from a CSV file, skipping the header row
# The file is named 'scorepdb.csv'
loss = np.loadtxt('scorepdb.csv', delimiter=',', skiprows=1)

# Generate epoch values linearly spaced from 0 to 50
epochs = np.linspace(0, 50, num=len(loss))

# Spline interpolation for smoothing
# Increasing 'k' or 's' might be necessary depending on data characteristics
spline = make_interp_spline(epochs, loss, k=9)  # k is the degree of the spline
smooth_epochs = np.linspace(epochs.min(), epochs.max(), 300)  # More points for a smoother curve
smooth_loss = spline(smooth_epochs)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(smooth_epochs, smooth_loss, label='loss', linestyle='-',linewidth=3, color='black')
#plt.scatter(epochs, loss, color='red', label='Original Data')  # Optionally plot original data points
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.grid(False)
plt.legend(fontsize=15)
plt.xticks(fontsize=15)  # Sets the font size of the x-tick labels
plt.yticks(fontsize=15)  # Sets the font size of the y-tick labels

plt.title('Evaluation Loss vs Epoch',fontsize=20)
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Evaluation Loss', fontsize=20)
#plt.legend()


# Save the plot to a file
plt.savefig('smoothed_evaluation_loss_vs_epoch.png', dpi=600,  bbox_inches='tight')
# Optionally display the plot
# plt.show()
