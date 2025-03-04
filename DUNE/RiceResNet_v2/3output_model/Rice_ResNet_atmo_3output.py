# trying to get list of url list to work 
import matplotlib.pylab as plt
import numpy as np
import zlib
import glob
import random
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers, callbacks, losses
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import keras.backend as K
import os
from generator_class_atmo_3output import DataGenerator_3output_train
from tensorflow.keras.optimizers import SGD
import json
import tqdm
import argparse
import pickle
import pandas as pd
from tensorflow.keras.regularizers import l2
from weighted_scce import WeightedSCCE #custom weighted loss function for class imbalance. 


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
    parser.add_argument('--listname', type=str, default='my_list.pkl', help='preprocessed data file name (df too)')
    ### CHANGE PATH 

    args = parser.parse_args()
    
    n_channels = 3
    dimensions = (args.pixel_map_size, args.pixel_map_size)
    params = {'batch_size':args.batch_size,'dim':dimensions, 'n_channels':n_channels}
    ### CHANGE PATH
    path_to_df = args.listname
    df = pd.read_pickle(path_to_df)
    generator = DataGenerator_3output_train(df, **params)
    
    ### CHANGE PATH
    history_filename = args.test_name+'_training_history.json'

    for dff in [df]:
        dff['pions'] = np.clip(dff['pions'], 0, 1) 
    
    partition = {'train': df.iloc[:int(.85*len(df))], 'validation': df.iloc[int(.85*len(df)):]}
    print(f"Number of pixel maps for training {len(partition['train'])} and for validation {len(partition['validation'])}")

    #==============================================
    # Model 
    #==============================================
    print('building model now.')
    input_shape = (dimensions[0], dimensions[1], n_channels)
    inputs = layers.Input(shape=input_shape)

    # Initial Convolution Layer
    x = layers.Conv2D(32, 7, strides=2, padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
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
    
    # Output layer, 3 classifications (no nu_tau) plus sub-class cases. 
    output_names=['flavour','protons','pions']
    output_neurons=[3,4,2]
    
    outputs = [None]*len(output_names)
    for i in range(len(outputs)):
        activation='sigmoid' if output_neurons[i]==1 else 'softmax'
        weight_decay = 1e-4
        outputs[i] = layers.Dense(output_neurons[i], use_bias=False, kernel_regularizer=l2(weight_decay),
                           activation=activation, name=output_names[i])(x)

  
    #class_weights = [{0: 1.4837743812022357, 1: 1.0, 2: 1.1844795010129983},
    #                 {0: 1.3145238622114037, 1: 1.0, 2: 3.9057891605873696, 3: 7.353197760094312},
    #                 {0: 1.0, 1: 3.5741821962728944, 2: 9.080158488265772, 3: 8.89313432835821},
    #                 ]

    #more specific to 3 output problem 
    class_weights = [{0: 1.48/(1.48+1+1.18), 1: 1./(1.48+1+1.18), 2: 1.18/(1.48+1+1.18)},
                     {0: 1.31/(1.31+1+3+5), 1: 1./(1.31+1+3+5), 2: 3./(1.31+1+3+5), 3: 5./(1.31+1+3+5)},
                     {0: 1./(1.+2), 1: 2./(1+2.)},
                     ]
    
    class_weights_tensors = [tf.constant(list(weights.values()), dtype=tf.float32) for weights in class_weights]

    bfce = losses.BinaryFocalCrossentropy(alpha=0.5, gamma=2, apply_class_balancing=True)
    
    output_losses = {
    "flavour": WeightedSCCE(class_weight=class_weights_tensors[0]), # "sparse_categorical_crossentropy",
    "protons":  WeightedSCCE(class_weight=class_weights_tensors[1]),
    "pions":  WeightedSCCE(class_weight=class_weights_tensors[2]),
    }

    output_loss_weights = {"flavour": 1.0, "protons": 1.0, "pions": 1.0, }
    # Create the model
    model = models.Model(inputs, outputs)
    
    # Define the learning rate scheduler callback and history saver
    lr_scheduler = LearningRateSchedulerPlateau(factor=0.5, patience=5, min_lr=1e-6)
    history_saver = SaveHistoryToFile(history_filename)

    early_stopping = EarlyStopping(monitor="val_flavour_loss", min_delta=0, patience=5, 
                               verbose=0,mode="auto",baseline=None,
                               restore_best_weights=True,
                                   )
    
    train_generator = DataGenerator_3output_train(partition['train'], **params)
    validation_generator = DataGenerator_3output_train(partition['validation'], **params)
    sgd_optimizer = SGD(learning_rate=args.learning_rate, momentum=0.9)
    
    model.compile(optimizer=sgd_optimizer,
                  loss = output_losses,
                  loss_weights = output_loss_weights,
              metrics=['accuracy', 'accuracy', 'accuracy'])
    
    model.summary()
    #Saving model summary
    with open(args.test_name+'_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    print('model compiled. training...') 
        
    history = model.fit(train_generator,validation_data=validation_generator,
                        epochs=args.num_epochs, 
                        callbacks=[lr_scheduler, history_saver
                                  ]) # no early stopping!! 

             
    #Save model
    model_name = 'saved_model/'+args.test_name
    model_name_keras = args.test_name+'.keras'
    model.save(model_name_keras)

    model.export(model_name)
