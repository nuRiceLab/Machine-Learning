# trying to get list of url list to work 
import numpy as np
import zlib
import glob
import random
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers, callbacks, losses
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import keras.backend as K
import os
from tensorflow.keras.optimizers import SGD
import json
import tqdm
import argparse
import pickle
import pandas as pd
from tensorflow.keras.regularizers import l2
from generator_class import DataGenerator

class LearningRateSchedulerPlateau(callbacks.Callback):
    '''
    Learning rate scheduler
    '''
    def __init__(self, factor=0.5, patience=2, min_lr=1e-6):
        super(LearningRateSchedulerPlateau, self).__init__()
        self.factor = factor          # Factor by which the learning rate will be reduced
        self.patience = patience      # Number of epochs with no improvement after which learning rate will be reduced
        self.min_lr = min_lr          # Minimum learning rate allowed
        self.wait = 0                 # Counter for patience
        self.best_val_loss = 1e5        # Best validation accuracy

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        if current_val_loss is None:
            return

        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.wait = 0
                new_lr = tf.keras.backend.get_value(self.model.optimizer.lr) * self.factor
                new_lr = max(new_lr, self.min_lr)
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                print(f'\nLearning rate reduced to {new_lr} due to plateau in validation loss.')

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

def residual_block(x, filters, kernel_size=3, stride=1):
    # Shortcut connection
    shortcut = x

    # First convolution layer
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second convolution layer
    x = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)

    # If the number of filters has changed, apply a 1x1 convolution to the shortcut
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same', kernel_initializer='he_normal')(shortcut)
    
    # Add the shortcut to the output
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def get_pred_class(pred_scores):
    if pred_scores.shape[1] == 1: 
        pred_class = np.int32(pred_scores+.5) #messy I know but they do 0-0.99 as 0 and 1-1.99 as 1. 
    else: 
        pred_class = np.argmax(pred_scores, axis=1)
    return pred_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--pixel_map_size', type=int, help='Pixel map size square shape')
    parser.add_argument('--test_name', type=str, help='name of model and plots')
    # parser.add_argument('--is_preprocessed', type=bool, default=False, help='is the data already preprocessed')
    parser.add_argument('--listname', type=str, default='None', help='preprocessed data file name (df too)')
    ### CHANGE PATH 
    parser.add_argument('--path_checkpoint', type=str, default='/home/', help='where you want to save the model after epoch, also where to load the weights before training. ')

    args = parser.parse_args()
    
    n_channels = 3
    dimensions = (args.pixel_map_size, args.pixel_map_size)
    params = {'batch_size':args.batch_size,'dim':dimensions, 'n_channels':n_channels}

    ### CHANGE PATH
    path_to_df = args.listname
    ### CHANGE PATH
    path_to_test_df = args.listname
    df = pd.read_pickle(path_to_df)
    dftest = pd.read_pickle(path_to_test_df)

    generator = DataGenerator(df, **params)
    test_generator = DataGenerator(dftest, **params)
    
    partition = {'train': df.iloc[:int(.83*len(df))], 'validation': df.iloc[int(.83*len(df)):]}
    print(f"Number of pixel maps for training {len(partition['train'])} and for validation {len(partition['validation'])}")

    ### CHANGE PATH
    history_filename = args.test_name+'_training_history.json'
    #==============================================
    # Model 
    #==============================================
    print('building model now.')
    input_shape = (dimensions[0], dimensions[1], n_channels)
    inputs = layers.Input(shape=input_shape)

    # Initial Convolution Layer
    x = layers.Conv2D(32, 7, strides=2, padding='same', kernel_initializer='he_normal', name='layer1')(inputs)
    x = layers.BatchNormalization(name='batch_norm1')(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Residual Blocks
    num_blocks = 5  # Increase the number of residual blocks
    filters = [32, 64, 128, 256, 512]   # Increase the number of filters in each block

     
    for i in range(num_blocks):
        x = residual_block(x, filters[i])

    # Global Average Pooling Layer
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected (Dense) layers
    x = layers.Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dense(64, activation='sigmoid', kernel_initializer='he_normal')(x)
    
     # Output layer with 3 units for classification
    outputs = layers.Dense(3, activation='softmax', name='output')(x)


    # Create the model
    model = models.Model(inputs, outputs)
    #checkpoint_filepath = args.path_checkpoint
    #checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=False)
    # Define the learning rate scheduler callback and history saver
    lr_scheduler = LearningRateSchedulerPlateau(factor=0.5, patience=5, min_lr=1e-6)
    history_saver = SaveHistoryToFile(history_filename)

    train_generator = DataGenerator(partition['train'], **params)
    validation_generator = DataGenerator(partition['validation'], **params)
    sgd_optimizer = SGD(learning_rate=args.learning_rate, momentum=0.9)
    
    model.compile(optimizer=sgd_optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
    model.summary()
    
    #Saving model summary
    with open(args.test_name+'_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    history = model.fit(train_generator,validation_data=validation_generator,
                        epochs=args.num_epochs, callbacks=[lr_scheduler, history_saver])

    model.save('my_model.keras')       
    model.export('saved_model/my_model')
