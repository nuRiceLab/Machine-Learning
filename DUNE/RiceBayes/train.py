import os
import json
import glob
import random
import argparse
import numpy as np

from BNN_model import bayes_model
from generator_class import DataGenerator

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers, callbacks
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers.legacy import SGD

#GPU/CPU Selection
gpu_setting = 'y'


def get_data(pixel_map_dir, generator):
    '''
    Get pixels maps 
    '''
    file_list_all = glob.glob(pixel_map_dir)
    file_list = []

    for f in file_list_all:
        if generator.get_info(f)['NuPDG'] != 16 and generator.get_info(f)['NuPDG'] != -16 and generator.get_info(f)['NuEnergy'] < 4.0:
            file_list.append(f)

    random.shuffle(file_list)
    
    return file_list


class LearningRateSchedulerPlateau(callbacks.Callback):
    '''
    Learning rate scheduler
    '''
    def __init__(self, factor=0.5, patience=5, min_lr=1e-6):
        super(LearningRateSchedulerPlateau, self).__init__()
        self.factor = factor          # Factor by which the learning rate will be reduced
        self.patience = patience      # Number of epochs with no improvement after which learning rate will be reduced
        self.min_lr = min_lr          # Minimum learning rate allowed
        self.wait = 0                 # Counter for patience
        self.best_val_acc = -1        # Best validation accuracy

    def on_epoch_end(self, epoch, logs=None):
        current_val_acc = logs.get('val_accuracy')
        if current_val_acc is None:
            return

        if current_val_acc > self.best_val_acc:
            self.best_val_acc = current_val_acc
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.wait = 0
                new_lr = tf.keras.backend.get_value(self.model.optimizer.lr) * self.factor
                new_lr = max(new_lr, self.min_lr)
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                print(f'\nLearning rate reduced to {new_lr} due to plateau in validation accuracy.')

class SaveHistoryToFile(callbacks.Callback):
    '''
    Save history to a file
    '''
    def __init__(self, file_path):
        super(SaveHistoryToFile, self).__init__()
        self.file_path = file_path
        self.history = {}
    def on_epoch_end(self, epoch, logs=None):
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
        
        with open(self.file_path, 'w') as file:
            json.dump(self.history, file)
   
 
def nll(y_true, y_pred):
    """
    The log-likelihood of a sample y_true given a predicted distribution y_pred 
    measures how probable the true sample is according to the predicted distribution. 
    It's a way of assessing how well the predicted distribution aligns with the actual data.
    By taking the negative of the log-likelihood, the measure is turned into a loss: higher values of this
    loss correspond to worse predictions (i.e., the true data being less likely under the predicted distribution).
    Minimizing this loss during training encourages the model to adjust its parameters to make the true data more
    likely under the predicted distribution
    """
    return -y_pred.log_prob(y_true)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--pixel_map_size', type=int, default=200, help='Pixel map size square shape')
    parser.add_argument('--pixel_maps_dir', type=str, help='Pixel maps directory')
    parser.add_argument('--test_name', type=str, default='test', help='name of model and plots')
    args = parser.parse_args()
    
    n_channels = 3
    dimensions = (args.pixel_map_size, args.pixel_map_size)
    params = {'batch_size':args.batch_size,'dim':dimensions, 'n_channels':n_channels}
    
    _files = glob.glob(args.pixel_maps_dir)
    generator = DataGenerator(_files, **params)
    # prepare data
    data = get_data(args.pixel_maps_dir, generator)
    
    print(f'Number of pixel maps for training {len(data)*0.9} and for validation {len(data)*0.1}')
    partition = {'train': data[:int(.9*len(data))], 'validation': data[int(.9*len(data)):]}
    
    #==============================================
    # Model 
    #==============================================
    input_shape = (dimensions[0], dimensions[1], n_channels)
    
    model = bayes_model(input_shape)
    
    model.summary()
    sgd_optimizer = SGD(learning_rate=args.learning_rate, momentum=0.9)

    model.compile(optimizer=sgd_optimizer, loss=nll, metrics=['accuracy'])
    
    checkpoint_filepath = '/Users/aaronhiguera/HEP/DUNE/Machine-Learning/DUNE/RiceBayes/'
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=False)
    
    # Define the learning rate scheduler callback and history saver
    lr_scheduler = LearningRateSchedulerPlateau(factor=0.5, patience=5, min_lr=1e-6)
    history_filename = args.test_name+'_training_history.json'
    history_saver = SaveHistoryToFile(history_filename)

    train_generator = DataGenerator(partition['train'], **params)
    validation_generator = DataGenerator(partition['validation'], **params)    
    
    model.fit(train_generator,validation_data=validation_generator,
              epochs=args.num_epochs, callbacks=[lr_scheduler, history_saver])

    # for inferences need to save weights
    weights = args.test_name+'.h5'
    model.save_weights(weights)
    #complete_model = args.test_name
    #model.save(complete_model)
