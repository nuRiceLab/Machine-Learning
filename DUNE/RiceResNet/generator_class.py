import numpy as np
import tensorflow as tf
from tensorflow import keras
import zlib
import glob

class DataGenerator(keras.utils.Sequence):
    def __init__(self, files, batch_size, dim, n_channels):
        self.files = files
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels

    # __normalize__ method removed
    
    def get_info(self, file):
        with open(file, 'rb') as info_file:
            info = info_file.readlines()
            truth = {}
            truth['NuPDG'] = int(info[7].strip())
            truth['NuEnergy'] = float(info[1])
            truth['LepEnergy'] = float(info[2])
            truth['Interaction'] = int(info[0].strip()) % 4
        return truth
    
    def get_pixels_map(self, file_name):
        cells = self.dim[0]
        planes = self.dim[1]
        views = self.n_channels
        file = open(file_name, 'rb').read()
        pixels_map = np.frombuffer(zlib.decompress(file), dtype=np.uint8)
        pixels_map = pixels_map.reshape(views, planes, cells)
        return pixels_map
    
    def get_data_and_labels(self, files):
        data_maps = []
        data_labels = []
        for file in files:
            pdg = abs(self.get_info(file)['NuPDG'])
            if pdg == 1:
                truth_label = 0
            elif pdg == 12:
                truth_label = 1
            elif pdg == 14:
                truth_label = 2
            elif pdg == 16:
                truth_label = 3
            image = file[:-5] + '.gz'
            data_maps.append(self.get_pixels_map(image))
            data_labels.append(truth_label)
        return data_maps, data_labels
    
    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = list(range(index * self.batch_size, (index + 1) * self.batch_size))
        
        files_temp = [self.files[i] for i in indexes]
        
        maps_temp, labels_temp = self.get_data_and_labels(files_temp)
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
        labels_temp = np.array(labels_temp)
        X, y = self.__data_generation(train_temp, labels_temp)
        return X, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.files))
    
    def __data_generation(self, train_temp, labels_temp):
        return train_temp, labels_temp





