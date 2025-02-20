import os
import glob
import argparse
import pandas as pd
from tqdm import tqdm
from generator_class import DataGenerator

def get_info(file):
    with open(file, 'rb') as info_file:
        info = info_file.readlines()
        truth = {}
        truth['NuPDG'] = int(info[7].strip())
        truth['NuEnergy'] = float(info[1])
        truth['LepEnergy'] = float(info[2])
        truth['Interaction'] = int(info[0].strip()) % 4
        return truth

def get_data(pixel_map_dir):
    '''
    Get pixels maps 
    '''
    map_dir = pixel_map_dir+'/*.info'
    file_list_all = glob.glob(map_dir)
    file_list = []    
    for f in tqdm(file_list_all):
        if get_info(f)['NuPDG'] != 16 and get_info(f)['NuPDG'] != -16:
            file_list.append(f)

    return file_list

def main():
    parser = argparse.ArgumentParser(description="Generate dataset and save it as a pickle file.")
    
    parser.add_argument(
        "--data_path", type=str, required=True, 
        help="Path to the input files (e.g., '/path/to/files/')"
    )
    
    parser.add_argument(
        "--output_pickle", type=str, required=True, 
        help="Path to save the output pickle file (e.g., 'output.pkl')"
    )

    args = parser.parse_args()

    print("Starting the process...")
    print(f"Loading data from: {args.data_path}")
    
    file_list = get_data(args.data_path)

    df_dummy = pd.DataFrame()  # Just to initialize

    n_channels = 3
    dimensions = (300, 300)
    params = {'batch_size': 2, 'dim': dimensions, 'n_channels': n_channels}

    generator = DataGenerator(df_dummy, **params)

    df = generator.get_dataframe(file_list)

    print(f"Saving dataframe to: {args.output_pickle}")
    pd.to_pickle(df, filepath_or_buffer=args.output_pickle)

if __name__ == "__main__":
    main()
