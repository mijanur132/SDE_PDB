import re
import sys
import os
import pandas as pd
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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_log_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    epochs, steps, losses = read_and_process_log(file_path)
    
    # Generate output file path
    output_dir, file_name = os.path.split(file_path)
    output_file_name = os.path.splitext(file_name)[0] + '_plot.png'
    output_path = os.path.join(output_dir, output_file_name)
    
    plot_losses(epochs, losses, output_path, window_size=100)
