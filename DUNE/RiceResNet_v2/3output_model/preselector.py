import os
import glob
import argparse
import pandas as pd
from tqdm import tqdm
from generator_class_atmo_3output import DataGenerator_3output_train
import numpy as np

def get_info(file):
    try:
        gz_file = file.replace('.info', '') + '.gz'
        if not os.path.exists(gz_file):
            raise FileNotFoundError(f"Missing required file: {gz_file}")
        
        with open(file, 'rb') as info_file:
            info = info_file.readlines()
            truth = {}
            truth['NuPDG'] = int(info[7].strip())
            truth['NuEnergy'] = float(info[1])
            truth['LepEnergy'] = float(info[2])
            truth['Interaction'] = int(info[0].strip()) % 4
            truth['NProton'] = int(info[8].strip())
            truth['NPion'] = int(info[9].strip())
            truth['NPizero'] = int(info[10].strip())  
            truth['NNeutron'] = int(info[11].strip())
            truth['is_antineutrino'] = int(int(info[7].strip())<0)
            truth['image_path'] = file[:-5] + '.gz'

            # extra labels for ML classification
            pdg = np.abs(truth['NuPDG'])
            if pdg == 1: truth['flavour'] = 0 #NC
            elif pdg == 12: truth['flavour'] = 1 #CC nu_e
            elif pdg == 14: truth['flavour'] = 2 #CC nu_mu
            elif pdg == 16: truth['flavour'] = 3 #CC nu_tau (not included in our E range but still here) 
            truth['protons'] = np.clip(truth['NProton'], None, 3) # 0,1,2,or N Protons 
            truth['pions'] = np.clip(truth['NPion'], None, 3) # 0,1,2,or N Pions 
            truth['pizeros'] = np.clip(truth['NPizero'], None, 3) # 0,1,2,or N Pizeros
            truth['neutrons'] = np.clip(truth['NNeutron'], None, 3) # 0,1,2,or N Neutrons 

            return truth
    except Exception as e:
        print(f"Error processing file: {file}, skipping. Error: {e}")
        return None  # Return None to indicate an issue with this file

def get_data(pixel_map_dir):
    '''
    Get pixels maps 
    '''
    map_dir = pixel_map_dir+'/*.info'
    file_list_all = glob.glob(map_dir)
    file_list = []    
    for f in tqdm(file_list_all):
        info = get_info(f)
        if info is not None:
            if get_info(f)['NuPDG'] != 16 and get_info(f)['NuPDG'] != -16 and get_info(f)['NuEnergy'] <20.0 :
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

    generator = DataGenerator_3output_train(df_dummy, **params)

    df = generator.get_dataframe(file_list)

    print(f"Saving dataframe to: {args.output_pickle}")
    pd.to_pickle(df, filepath_or_buffer=args.output_pickle)

if __name__ == "__main__":
    main()
