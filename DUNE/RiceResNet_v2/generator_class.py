import numpy as np
import tensorflow as tf
from tensorflow import keras
import concurrent.futures
import zlib
import glob
import pickle
import multiprocessing
from tqdm import tqdm
import pandas as pd


class DataGenerator(keras.utils.Sequence):
    '''
    going to assume you've already preselected the energy ROI. 
    can run get_data if you haven't yet and save to pickle file the results. 
    '''
    
    def __init__(self, df, batch_size, dim, n_channels):
        # self.files = files
        self.df = df
        self.df_length = len(self.df)
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.indexes = np.arange(self.df_length)
        np.random.shuffle(self.indexes)
    # __normalize__ method removed
    
    def get_info(self, file):
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
            
        return truth
    
    def get_dataframe(self, files):
        data_labels = {}
        for i, file in enumerate(tqdm(files)):
            truth_info = self.get_info(file)
            if i==0:
                for key in truth_info.keys():
                    data_labels[key] = [truth_info[key]]
            else: 
                for key in truth_info.keys():
                    data_labels[key].append(truth_info[key])
        df = pd.DataFrame.from_dict(data_labels)
        return df

    def save_dataframe(self, df, path_to_df):
        pd.to_pickle(df, filepath_or_buffer=path_to_df)

    def load_dataframe(self, path_to_df, idx=None):
        df = pd.read_pickle(path_to_df)
        return df 
        
    def get_pixels_map(self, file_name):
        cells = self.dim[0]
        planes = self.dim[1]
        views = self.n_channels
        f = open(file_name, 'rb')
        file = f.read()
        pixels_map = np.frombuffer(zlib.decompress(file), dtype=np.uint8)
        f.close()
        pixels_map = pixels_map.reshape(views, planes, cells)
        return pixels_map
    
    def get_data_and_labels(self, idxs):
        data_subset = self.df.iloc[idxs]
        # clip it so it's a simpler problem for the model to solve. 
        
        # data_subset = self.df.iloc[idxs[0]:idxs[1]]
        labels = ['flavour']
        data_label_dict = data_subset[labels].to_numpy()
        # Use ThreadPoolExecutor for parallel file reading
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Map the file reading function to each file in parallel
            data_maps = list(executor.map(self.get_pixels_map, data_subset['image_path']))
        return data_maps, data_label_dict

    
    def __len__(self):
        return int(np.floor(self.df_length / self.batch_size))
    
    def __getitem__(self, index):
        # Use the shuffled indices
        upper_point = min((index + 1) * self.batch_size, self.df_length)
        indexes = self.indexes[index*self.batch_size:upper_point]

        maps_temp, labels_temp = self.get_data_and_labels(indexes)
        maps_z_view = np.asarray(maps_temp)[:, 2:]
        maps_v_view = np.asarray(maps_temp)[:, 1:2]
        maps_u_view = np.asarray(maps_temp)[:, 0:1]

        train_temp = []
        if self.n_channels == 1:
            for i in range(len(maps_z_view)):
                train_temp.append(maps_z_view[i][0])

        elif self.n_channels == 3:
            for i in range(len(maps_z_view)):
                train_temp.append(np.dstack((maps_u_view[i][0], maps_v_view[i][0], maps_z_view[i][0])))

        train_temp = np.array(train_temp).reshape([self.batch_size, self.dim[0], self.dim[1], self.n_channels])
        
        # labels_temp is a dictionary containing multiple output labels
        X, y = self.__data_generation(train_temp, labels_temp)
        return X, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # Shuffle indices at the end of each epoch
        np.random.shuffle(self.indexes)
    
    def __data_generation(self, train_temp, labels_temp):
        return train_temp, labels_temp