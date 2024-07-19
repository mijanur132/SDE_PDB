import numpy as np
import os
import sys

def split_and_save_npy_files(input_directory, output_directory):
    # Ensure the output directory exists, create if it doesn't
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory at {output_directory}")

    # List all files in the input directory
    files = [f for f in os.listdir(input_directory) if f.endswith('.npy')]
    
    # Iterate over each file
    for file in files:
        print(file)
        file_path = os.path.join(input_directory, file)
        try:
            # Load the numpy array from the .npy file
            data = np.load(file_path)
        except IOError as e:
            print(f"Error loading {file_path}: {e}. Skipping file.")
            continue
        
        # Calculate the number of chunks
        num_chunks = data.shape[0] // 40
        if data.shape[0] % 40 != 0:
            num_chunks += 1
        
        chunks = np.array_split(data, num_chunks, axis=0)
        
        for i, chunk in enumerate(chunks):
            new_file_name = f"{os.path.splitext(file)[0]}_{i+1}.npy"
            new_file_path = os.path.join(output_directory, new_file_name)
            if chunk.size==0:
                print("chunk empty...")
                continue
            try:
                np.save(new_file_path, chunk)
                print(f"Saved chunk {i+1} with shape {chunk.shape} as {new_file_name}")
            except IOError as e:
                print(f"Failed to save {new_file_path}: {e}. Continuing with next chunk.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_directory> <output_directory>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    split_and_save_npy_files(input_directory, output_directory)
