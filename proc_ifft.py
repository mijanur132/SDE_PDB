import numpy as np
import os
import sys
import matplotlib.pyplot as plt

def main():

    if len(sys.argv) < 3:
        print("Usage: python script_name.py <input_directory> <output_directory>")
        sys.exit(1)

    # Directories from command line
    input_directory = sys.argv[1]
    output_directory=sys.argv[2]
    target_size = 320
    mini_batch_size=10

    # Process each .npy file in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".npy"):
            data_path = os.path.join(input_directory, filename)
            k_space_data = np.load(data_path)
            shifted_k_space = np.fft.fftshift(k_space_data, axes=(0,1, 2))
            truncated_data=shifted_k_space#[:-1,:-1,:-1] #assume the number of input bins is odd, remove highest freq to make it even in accordance with numpy freq convention
            n_trunc=(np.maximum(np.array(truncated_data.shape)-target_size,0))//2# number of removed bin on each side 
            truncated_data=truncated_data[n_trunc[0]:truncated_data.shape[0]-n_trunc[0], n_trunc[1]:truncated_data.shape[1]-n_trunc[1], n_trunc[2]:truncated_data.shape[2]-n_trunc[2]]
            
            padding_needed_first_dim = (mini_batch_size - truncated_data.shape[0] % mini_batch_size) % mini_batch_size
            if padding_needed_first_dim > 0:
                padding = [(0, padding_needed_first_dim)]  # Apply the calculated padding to the first dimension
            else:
                padding = [(0, 0)]  # No padding needed
            for dim in truncated_data.shape[1:]:  # Only pad spatial dimensions
                pad_size = (target_size - dim) // 2
                pad_size=max(pad_size,0)
                padding.append((pad_size,pad_size+1))  

            padded_k_space = np.pad(truncated_data, pad_width=padding, mode='constant')
            padded_k_space=np.fft.ifftshift(padded_k_space, axes=(0,1,2))
            # Apply inverse Fourier Transform to the entire volume
            image_volume = np.fft.ifftn(padded_k_space, axes=(0,1, 2))
            image_volume=np.real(image_volume)
            image_volume=image_volume/image_volume.std()
            plt.imsave(os.path.join(output_directory, f"{filename}_ifft_0th_slice.png"), np.abs(image_volume[0]), cmap='gray')

            #Save the processed complex volume to the output directory
            output_path = os.path.join(output_directory, filename)
            #np.save(output_path, image_volume)
            #print(f"Processed volume saved as {output_path} and shape:{image_volume.shape}")

              # Calculate the number of chunks
            num_chunks = image_volume.shape[0] // mini_batch_size
            if image_volume.shape[0] % mini_batch_size != 0:
                num_chunks += 1
            chunks = np.array_split(image_volume, num_chunks, axis=0)
            for i, chunk in enumerate(chunks):
                if chunk.size == 0:
                    print("chunk empty...")
                    continue
                #plt.imsave(os.path.join(output_directory, f"{filename}_ifft_0th_slice.png"), np.abs(image_volume[0]*255), cmap='gray')
                new_file_name = f"{os.path.splitext(filename)[0]}_{i+1}.npy"
                new_file_path = os.path.join(output_directory, new_file_name)
                try:
                    np.save(new_file_path, chunk)
                    print(f"Saved chunk {i+1} with shape {chunk.shape} as {new_file_name}")
                except IOError as e:
                    print(f"******************Failed to save {new_file_path}: {e}. Continuing with next chunk.")


if __name__ == "__main__":
    main()
