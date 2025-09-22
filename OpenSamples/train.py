import os
import json
import glob
import random
import argparse
import numpy as np
import pandas as pd

from BNN_model import bayes_model, bayes_three_tower
from DataGenerator import DataGenerator

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers.legacy import SGD

#GPU/CPU Selection
gpu_setting = 'y'


class LearningRateSchedulerPlateau(callbacks.Callback):
    '''
    Learning rate scheduler
    '''
    def __init__(self, factor=0.5, patience=5, min_lr=1e-4):
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
    return -y_pred.log_prob(tf.cast(y_true, tf.float32))


# example weights: inversely proportional to frequency
class_weights_vec = tf.constant([1.0/0.64, 1.0/0.09, 1.0/0.27], dtype=tf.float32)
class_weights_vec /= tf.reduce_sum(class_weights_vec) / tf.cast(tf.size(class_weights_vec), tf.float32)
def weighted_nll(y_true, y_pred_dist):
    """
    y_true: one-hot (B, C)
    y_pred_dist: tfp.distributions.OneHotCategorical
    """
    # per-sample weight = dot(y_true, class_weights)
    w = tf.reduce_sum(y_true * class_weights_vec, axis=-1)  # (B,)
    nll = -y_pred_dist.log_prob(tf.cast(y_true, tf.float32))  # (B,)
    return tf.reduce_mean(w * nll)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--pixel_map_size', type=int, default=500, help='Pixel map size square shape')
    parser.add_argument('--pixel_maps', type=str, help='Pre-selected pixel maps ')
    parser.add_argument('--test_name', type=str, default='test', help='name of model and plots')
    args = parser.parse_args()
    
    n_channels = 3
    dimensions = (args.pixel_map_size, args.pixel_map_size)
    params = {'batch_size':args.batch_size,'dim':dimensions, 'n_channels':n_channels}
    
    
    train_set = DataGenerator(
                        data_path=args.pixel_maps,
                        batch_size=args.batch_size,
                        shuffle=True,
                        channels_last=True,  # (B, 500, 500, 3) for typical Keras models
                        one_hot=True,        # <- make labels one-hot
                        num_classes=3,
                )
    val_path = args.pixel_maps +"/val/"
    val_set = DataGenerator(
                        data_path=val_path,
                        batch_size=args.batch_size,
                        shuffle=True,
                        channels_last=True,  # (B, 500, 500, 3) for typical Keras models
                        one_hot=True,        # <- make labels one-hot
                        num_classes=3,
                )
    print(f"training batches {len(train_set)}")
    tf.keras.backend.clear_session()
    #==============================================
    # Model 
    #==============================================
    input_shape = (dimensions[0], dimensions[1], n_channels)
    
    #model = bayes_model(input_shape)
    model = bayes_three_tower(input_shape)
    
    model.summary()

    optimizer = tf.keras.optimizers.Adam(args.learning_rate)

    model.compile(optimizer=optimizer,
              loss=weighted_nll,
              metrics=['accuracy'],
              jit_compile=True)   # XLA JIT for the train step
    
    checkpoint_filepath = '/home/dirac/HEP/OpenSamples/ML/Machine-Learning/OpenSamples/'
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,
                                          save_best_only=True, save_weights_only=True, monitor='val_loss')
    
    # Define the learning rate scheduler callback and history saver
    lr_scheduler = LearningRateSchedulerPlateau(factor=0.5, patience=5, min_lr=1e-6)
    history_filename = args.test_name+'_training_history.json'
    history_saver = SaveHistoryToFile(history_filename)
    early_stopper = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.001, mode='min',
                                  restore_best_weights=True)
    
    model.fit(train_set, validation_data=val_set, epochs=args.num_epochs,
              callbacks=[lr_scheduler, history_saver, early_stopper, checkpoint_callback],
              workers=24, use_multiprocessing=True, max_queue_size=8)
    
    # for inferences need to save weights
    weights = args.test_name+'.h5'
    model.save_weights(weights)
    ## !!!!!need to fix this!!!
    complete_model = args.test_name
    model.save(complete_model)
    