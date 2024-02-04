import h5py

# Replace 'your_file.h5' with the path to your HDF5 file
file_path = 'test_subset.h5'

# Open the file in read mode ('r')
with h5py.File(file_path, 'r') as hdf:
    # Now you can read datasets, attributes, etc., from the file
    all_datasets = list(hdf.keys())
    # For example, to read a dataset named 'dataset_name'
    #data = hdf['dataset_name'][:]
    print(all_datasets)

    for k in all_datasets:
        info = []
        pixel_map  = []
        if 'info' in k:
            info = hdf[k]
            print(info[1])
        elif 'gz' in k:
            pixel_map = hdf[k]
            print(pixel_map)

