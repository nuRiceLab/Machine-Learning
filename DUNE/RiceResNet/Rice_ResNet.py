import matplotlib.pylab as plt
import numpy as np
import zlib
import glob
import random
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers, callbacks
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from generator_class import DataGenerator
from tensorflow.keras.optimizers import SGD
import json
import argparse

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
    split = int(.9*len(file_list)) # 10% for testing
    allfiles, testfiles = file_list[:split], file_list[split:]
    
    return allfiles, testfiles


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


            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--pixel_map_size', type=int, help='Pixel map size square shape')
    parser.add_argument('--pixel_maps_dir', type=str, help='Pixel maps directory')
    parser.add_argument('--test_name', type=str, help='name of model and plots')
    args = parser.parse_args()
    
    n_channels = 3
    dimensions = (args.pixel_map_size, args.pixel_map_size)
    params = {'batch_size':args.batch_size,'dim':dimensions, 'n_channels':n_channels}
    
    _files = glob.glob(args.pixel_maps_dir)
    generator = DataGenerator(_files, **params)
    # prepare data
    data, testdata = get_data(args.pixel_maps_dir, generator)
    
    print(f'Number of pixel maps for training {len(data)*0.83} and for validation {len(data)*0.2}')
    partition = {'train': data[:int(.83*len(data))], 'validation': data[int(.83*len(data)):]}


    history_filename = args.test_name+'training_history.json'
    #==============================================
    # Model 
    #==============================================
    input_shape = (dimensions[0], dimensions[1], n_channels)
    inputs = layers.Input(shape=input_shape)

    # Initial Convolution Layer
    x = layers.Conv2D(32, 7, strides=2, padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Residual Blocks
    num_blocks = 3  # Increase the number of residual blocks
    filters = [32, 64, 128]   # Increase the number of filters in each block

    for i in range(num_blocks):
        x = residual_block(x, filters[i])

    # Global Average Pooling Layer
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected (Dense) layers
    x = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dense(32, activation='sigmoid', kernel_initializer='he_normal')(x)
    # Output layer with 3 units for classification
    outputs = layers.Dense(3, activation='softmax')(x)


    # Create the model
    model = models.Model(inputs, outputs)
    checkpoint_filepath = '/home/higuera/CNN/tmp_models/'
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=False)
    # Define the learning rate scheduler callback and history saver
    lr_scheduler = LearningRateSchedulerPlateau(factor=0.5, patience=5, min_lr=1e-6)
    history_saver = SaveHistoryToFile(history_filename)

    train_generator = DataGenerator(partition['train'], **params)
    validation_generator = DataGenerator(partition['validation'], **params)
    sgd_optimizer = SGD(learning_rate=args.learning_rate, momentum=0.9)
    
    model.compile(optimizer=sgd_optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
    
    #Saving model summary
    with open('/home/higuera/CNN/model_save/'+args.test_name+'_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
 
    history = model.fit(train_generator,validation_data=validation_generator,
                        epochs=args.num_epochs, callbacks=[lr_scheduler, history_saver])

             
    #Save model
    model_path = '/home/higuera/CNN/model_save/'+args.test_name
    model.save(model_path)
    print()
    print('Model saved to: ' + model_path)
    
    test_labels = generator.get_data_and_labels(testdata)[1]

    print('Checking test sample')
    num_labels = 3

    test_generator = DataGenerator(testdata, **params)
    test_loss, test_acc = model.evaluate(test_generator, verbose=4)
    predictions = model.predict(test_generator)
    predict = []
    for i in range(len(predictions)):
        predict.append(list(predictions[i]).index(max(list(predictions[i]))))
    con_mat = tf.math.confusion_matrix(np.asarray(test_labels[:len(predict)]), predict, num_classes = num_labels)
    con_mat_norm = [con_mat[i]/sum(con_mat[i])*100 for i in range(len(con_mat))]
    con_mat_norm = np.round(con_mat_norm, 2)
    plt.figure()
    for i in range(len(con_mat)):
        for j in range(len(con_mat[i])):
            plt.text(j-.2, i+.075, str(np.round(con_mat_norm[i][j], 1))+'%', c='k', backgroundcolor='w')
    plt.imshow(con_mat_norm)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    labels = ['NC', 'nu_e', 'nu_mu']
    plt.xticks([0, 1, 2], labels)
    plt.yticks([0, 1, 2], labels)
    plt.colorbar(label='Percent of True Values Predicted')
    plt.title('Normalized Confusion Matrix')
    plt.savefig('/home/higuera/CNN/plots/'+args.test_name+'_con_mat.pdf')