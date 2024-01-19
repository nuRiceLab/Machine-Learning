import numpy as np
import h5py
import os
import zlib
import argparse

def process_files(directory, output):
    # Create a new HDF5 file
    output_file = output+'.h5'
    with h5py.File(output_file, 'w') as hdf:
        # Iterate over files in the directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)

            # Check if the file is a .info file
            if filename.endswith('.info'):
                with open(file_path, 'r') as file:
                    numbers = [float(line.strip()) for line in file if line.strip()]
                    hdf.create_dataset(f'info_{filename}', data=numbers)

            # Check if the file is a .gz file
            if filename.endswith('.gz'):
                fmap = open(file_path, 'rb').read()
                pixels_map = np.frombuffer(zlib.decompress(fmap), dtype=np.uint8)
                pixels_map = pixels_map.reshape(3, 200, 200) # Adjust dimensions as needed       
                hdf.create_dataset(f'map_{filename}', data=pixels_map, compression="gzip", compression_opts=9)

def main():
    parser = argparse.ArgumentParser(description="Process files and store in HDF5 format.")
    parser.add_argument("directory", type=str, help="Directory containing the files to process")
    parser.add_argument("output", type=str, help="output file name")
    args = parser.parse_args()

    process_files(args.directory, args.output)

if __name__ == "__main__":
    main()

