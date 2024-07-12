import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <path_to_k_space_data.npy>")
        sys.exit(1)

    # Load the k-space data from the command line input
    data_path = sys.argv[1]
    k_space_data = np.load(data_path)
    print("Loaded k-space data shape:", k_space_data.shape)

    # Ask user for output directory
    save_directory = input("Please enter the output directory to save image slices: ")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        print(f"Directory created: {save_directory}")
    
    # Apply inverse Fourier Transform to each slice
    for i in range(k_space_data.shape[0]):
        # Convert k-space slice to image slice
        image_slice =np.abs(np.fft.ifft2(k_space_data[i]))
        #image_slice =np.abs(k_space_data[i]) 
        # Normalize the image slice for better visualization
        normalized_slice = (image_slice - np.min(image_slice)) / (np.max(image_slice) - np.min(image_slice))

        # Plot and save the image slice
        plt.figure(figsize=(6, 6))
        plt.imshow(normalized_slice, cmap='gray')
        plt.axis('off')  # Remove axis

        # Save each slice to the specified output directory
        slice_filename = f"slice_{i}.png"
        slice_path = os.path.join(save_directory, slice_filename)
        plt.savefig(slice_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the figure to free up memory
        print(f"Saved slice {i} as {slice_path}")

if __name__ == "__main__":
    main()

