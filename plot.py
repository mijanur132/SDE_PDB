import re
import sys
import os
# import pandas as pd
import matplotlib.pyplot as plt

def read_and_process_log(file_path):
    # Regex to match the relevant log entries
    pattern = r"epoch: (\d+), step: (\d+), training_loss: ([\d\.e\+\-]+)"
    
    epochs = []
    steps = []
    losses = []
    
    # Read the file
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                step = int(match.group(2))
                loss = match.group(3)
                # Convert loss to float, handling NaNs
                try:
                    loss = float(loss)
                except ValueError:
                    loss = None  # Convert bad values to None, which pandas handles as NaN
                epochs.append(epoch)
                steps.append(step)
                losses.append(loss)
    
    return epochs, steps, losses

def plot_losses(epochs, losses, output_path, window_size=10):
    # Convert losses to a pandas DataFrame
    data = pd.DataFrame({
        'Epoch': epochs,
        'Loss': losses
    })
    
    # Handle NaN values: option to fill or drop
    # data['Loss'].fillna(method='ffill', inplace=True)  # Forward fill
    data.dropna(inplace=True)  # Or drop all rows with NaN values

    # Calculate moving average
    data['Moving Average'] = data['Loss'].rolling(window=window_size).mean()
    print(data)
    plt.figure(figsize=(10, 5))
    #plt.plot(data['Epoch'], data['Loss'], label='Training Loss')
    plt.plot(data['Epoch'], data['Moving Average'], label='Moving Average', color='red')
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.ylim(0, 5)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.show()

def plot_loss_noise():

    labels = [0,2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100]
    loss_cleared = [1,1, 1, 0.99, 0.98, 0.97, 0.92, 0.89, 0.85, 0.83, 0.81, 0.79, 0.71, 0.48, 0.23, 0.21, 0.21, 0.21]
    loss2noisy_sample = [1,0.98, 0.81, 0.67, 0.57, 0.48, 0.38, 0.34, 0.25, 0.28, 0.26, 0.23, 0.16, 0.15, 0.13, 0.12, 0.11, 0.10]
    loss_clea_1=[1,0.99,0.99,0.99,0.97,0.95,0.91,0.89,0.85,0.77,0.74,0.69,0.32,0.44,0.25,0.21,0.14,0.18]
    loss_clea_5=[1,0.99,0.99,0.97,0.93,0.9,0.72,0.76,0.69,0.63,0.55,0.34,0.18,0.18,0.23,0.238,0.11,0.093]
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(labels, loss_cleared, marker='o', label='Cleaned Sample(0)', linestyle='-', linewidth='4')
    plt.plot(labels, loss2noisy_sample, marker='s', label='Noisy Sample', linestyle='--', linewidth='4')
    plt.plot(labels,loss_clea_1, label='Cleaned Sample(1)', linestyle='--', linewidth='4')
    plt.plot(labels, loss_clea_5, label='Cleaned Sample(-5)', linestyle='--', linewidth='4')

    # Adding title and labels
    #plt.title('Comparing Cleaned and Noisy Samples with Original Samples',fontsize=25, pad=20)


    plt.xticks( fontsize = 20, fontweight ='bold')  # Set custom ticks and labels

    plt.yticks(fontsize =20,fontweight='bold')
    # Adding title and labels
    #plt.title('Training Loss vs Steps', fontsize=25, fontweight = 'bold')
    plt.xlabel('Noise Level',fontsize=25, fontweight = 'bold')
    plt.ylabel('SSIM', fontsize=25, fontweight = 'bold')
    plt.ylim(0, 1.05)  # Set the maximum y value to 2

    # Adding legend
    plt.legend( fontsize='25', loc='upper right')
    plt.grid(visible= 'True', which= 'both', linestyle='dotted', linewidth=1 )
    plt.tight_layout()

    ax = plt.gca()

    # Set spine width
    spine_width = 4  # Define the width of the border
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)



    # Show the plot
    plt.savefig("/lustre/orion/stf218/proj-shared/brave/score-MRI/results/n-3pol/noise_plot.png")



if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python script.py <path_to_log_file>")
    #     sys.exit(1)

    plot_loss_noise()
    file_path = sys.argv[1]
    epochs, steps, losses = read_and_process_log(file_path)
    
    # Generate output file path
    output_dir, file_name = os.path.split(file_path)
    output_file_name = os.path.splitext(file_name)[0] + '_plot.png'
    output_path = os.path.join(output_dir, output_file_name)
    
    plot_losses(epochs, losses, output_path, window_size=100)
