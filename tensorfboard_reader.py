import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

import numpy as np

def moving_average(data, window_size):
    """ Compute the moving average of the given list using a window of the specified size. """
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')


# Path to your TensorBoard log file directory
log_file_dir = '/home/xrv/score-mri-palash/workdir/B/tensorboard/'
plot_save_path = os.path.join(log_file_dir, 'training_loss_vs_steps.png')

# Initialize an Event Accumulator with the path
ea = event_accumulator.EventAccumulator(
    log_file_dir,
    size_guidance={event_accumulator.SCALARS: 0},  # Load all scalar events
)

# Load all events from the file
ea.Reload()

# Ensure the tag 'training_loss' is present
if 'training_loss' in ea.Tags()['scalars']:
    # Retrieve the loss events
    training_loss_events = ea.Scalars('training_loss')
    
    # Extract steps and values for plotting
    steps = [event.step for event in training_loss_events]
    losses = [event.value for event in training_loss_events]
    print(steps)
    # Compute the moving average of losses
    window_size = 100  # Adjust the window size as needed
    losses_ma = moving_average(losses, window_size)
    
    # Plot the training losses and their moving average sequentially
    plt.figure(figsize=(30, 10))
    plt.plot(losses, label='Original Losses', marker='o', linestyle='-', color='b')
    plt.plot(range(window_size - 1, len(losses)), losses_ma, label='Moving Average', linestyle='-', color='r')
    plt.xlabel('Index')
    plt.ylabel('Training Loss')
    plt.ylim(2000, 10000)
    plt.title('Training Loss with Moving Average')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to the specified directory
    plt.savefig(plot_save_path)
    print(f"Training loss plot saved at {plot_save_path}")
else:
    print("The tag 'training_loss' is not found in the TensorBoard logs.")
