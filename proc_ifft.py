import numpy as np
import os
import sys
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <input_directory> <output_directory>")
        sys.exit(1)

    # Directories from command line
    input_directory = sys.argv[1]
    

    # Process each .npy file in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".npy"):
            data_path = os.path.join(input_directory, filename)
            k_space_data = np.load(data_path)
            #print(k_space_data.shape)
            #k_space_data=k_space_data.transpose(2,1,0)

            shifted_k_space = np.fft.fftshift(k_space_data, axes=(0,1, 2))

            print("shifted k space:",shifted_k_space[shifted_k_space>0])

            truncated_data=shifted_k_space[:-1,:-1,:-1]

            print(truncated_data[truncated_data>0])

            target_size = 320

            print(f"Loaded {filename} with shape: {k_space_data.shape}")
            print(k_space_data[k_space_data>0])
            n_trunc=(np.maximum(np.array(truncated_data.shape)-target_size,0))//2
            print("ntrunc:",n_trunc)
            truncated_data=truncated_data[n_trunc[0]:truncated_data.shape[0]-n_trunc[0], n_trunc[1]:truncated_data.shape[1]-n_trunc[1], n_trunc[2]:truncated_data.shape[2]-n_trunc[2]]
            print(truncated_data[truncated_data>0])
            print(truncated_data.shape)
            # Save the 10th slice of original k_space_data
            #plt.imsave(os.path.join(input_directory, f"{filename}_original_10th_slice.png"), np.abs(k_space_data[10]), cmap='gray')

    
            # Save the 10th slice after FFT shift
            #plt.imsave(os.path.join(output_directory, f"{filename}_fftshift_10th_slice.png"), np.abs(shifted_k_space[10]), cmap='gray')

            # Define padding sizes based on each dimension's size, adding 20% more padding
            #padding = [(0, 0)] + [(int(5 * dim), int(5* dim)) for dim in shifted_k_space.shape[1:]]

            
            padding = []  # No padding for the first dimension (depth)
            for dim in truncated_data.shape:  # Only pad spatial dimensions
                pad_size = (target_size - dim) // 2
                pad_size=max(pad_size,0)
                padding.append((pad_size,pad_size))  

            print(padding)
            padded_k_space = np.pad(truncated_data, pad_width=padding, mode='constant')
            print(padded_k_space[padded_k_space>0])
       
            # Save the 10th slice after padding
            #plt.imsave(os.path.join(output_directory, f"{filename}_padded_10th_slice.png"), np.abs(padded_k_space[10]), cmap='gray')

            padded_k_space=np.fft.ifftshift(padded_k_space, axes=(0,1,2))

            print(padded_k_space[padded_k_space>0])
       
            
            # Apply inverse Fourier Transform to the entire volume
            image_volume = np.fft.ifftn(padded_k_space, axes=(0,1, 2))
            print(image_volume.shape)

            # Save the 10th slice after IFFT
            plt.imsave(os.path.join(input_directory, f"{filename}_ifft_10th_slice.png"), np.abs(image_volume[10]), cmap='gray')

            # Save the processed complex volume to the output directory
            newpath=os.path.join(input_directory,"processed")
            if not os.path.exists(newpath):
                os.mkdir(newpath)
            output_path = os.path.join(newpath, filename)
            np.save(output_path, image_volume)
            print(f"Processed volume saved as {output_path}")

if __name__ == "__main__":
    main()
