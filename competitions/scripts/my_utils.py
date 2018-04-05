import time
import os, shutil
import zipfile

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

# Methods:

def bin_dataset(base_dir, sub_dir, mapping, validation_split = 0.0, labels = [], random_state = 0):
    """Bins the dataset into respective classes and returns a paths to the 'binned' directories.
       Complements the 'ImageDataGenerator' class in Keras.
       The train/val split mappings are stored into '.csv' files.

    # Arguments
        base_dir: Base directory
        sub_dir: Name of sub-directory, you want 'binned'.
        mapping: Numpy array of shape (N,2). First column -> filenames/id; Second column  -> labels. 
        [labels]: Class labels. Minimum of 2 needed. If 'empty', inferred from the mapping. 
        [validation_split]: Used to split the 'dataset' into training and validation sets.
        [random_state] : Random seed for train/val split.

    # Returns
        paths -> (base_dir/bin/train, base_dir/bin/val)
    
    """
    if labels == []: labels = np.unique(mapping[:,1])
    _dir = os.path.join(base_dir, sub_dir)
    _has_files = any([os.path.isfile(os.path.join(_dir, fname)) for fname in os.listdir(_dir)])

    # Checks
    assert os.path.isdir(_dir), "Error: The directory doesn't exist."
    assert _has_files, "Error: No files found."
    assert mapping.shape == mapping.reshape(-1,2).shape, "Error: The mapping can't be resolved."
    assert labels.size > 1, "Error: The size of 'labels' needs to be strictly > 1."
    assert validation_split in np.arange(0.0,1.0,1e-3), "Error: Invalid argument value: 'validation_split'"
    print('Tests passed.')

    print('\nThis might take a while...')
    start = time.time()

    print('Total no. of files: ', )
    print('Selected no. of files: ', len(mapping))

    # Helper functions
    def create_labels_folder(directory, labels):
        for label in labels:    
            path = os.path.join(directory,label)
            if not os.path.isdir(path): os.makedirs(path)
    
    def copy_files_into_labels_folders(src_dir, dst_dir, mapping):
        for fname, label in mapping.items():
            fname += '.jpg'
            src = os.path.join(src_dir,fname)
            dst = os.path.join(dst_dir, os.path.join(label,fname))
            if os.path.isfile(src): shutil.copy(src,dst)
    
    # Directory paths
    bin_dir = os.path.join(base_dir, 'bin')
    if not os.path.isdir(bin_dir): os.mkdir(bin_dir)
    train_path = os.path.join(bin_dir, 'train')
    val_path = os.path.join(bin_dir, 'val')

    # Split and bin the files into training and validation sets.
    if validation_split > 0.0:
        # Computing the split
        X, y = mapping[:,0], mapping[:,1]
        X_train, X_val, y_train, y_val = train_test_split( 
                                        X, y,
                                        test_size = validation_split, 
                                        random_state=random_state, 
                                        stratify = y,
                                        )
        # Need to reshape the numpy arrays to use them
        X_train = X_train.reshape(-1,1)
        y_train = y_train.reshape(-1,1)
        X_val = X_val.reshape(-1,1)
        y_val = y_val.reshape(-1,1)

        train_mapping = np.concatenate([X_train,y_train], axis = 1)
        val_mapping = np.concatenate((X_val,y_val), axis = 1)

        # Binning the files
        create_labels_folder(train_path, labels)
        create_labels_folder(val_path, labels)
        copy_files_into_labels_folders(_dir, train_path, dict(train_mapping))
        copy_files_into_labels_folders(_dir, val_path, dict(val_mapping))

        # Saving mappings to '.csv' files
        data_tmap = pd.DataFrame(train_mapping, columns = ['id','label'])
        data_vmap = pd.DataFrame(val_mapping, columns = ['id','label'])
        data_tmap.to_csv(os.path.join(bin_dir, 'train_map.csv'))
        data_vmap.to_csv(os.path.join(bin_dir, 'val_map.csv'))

        print("No. of train samples:" ,train_mapping.shape[0])
        print("No. of validation samples:" ,val_mapping.shape[0])
    
    else:
        # Binning the files
        create_labels_folder(train_path, labels)
        copy_files_into_labels_folders(_dir, train_path, dict(mapping))

    end = time.time()
    print('Done. Time taken: %.2fs.' %(end - start))

    return train_path, val_path


def unpack_dataset(src_dir, dest_dir):
    """Unpacks files in the 'src_dir' to 'dest_dir'.
    """
   # Checks
    assert os.path.isdir(src_dir), "Error: {} doesn't exist.".format(src_dir)
    assert os.path.isdir(dest_dir), "Error: {} doesn't exist.".format(dest_dir)
    print('Tests passed.')

    print("Extracting from '{}' to '{}'.".format(src_dir,dest_dir))
    print('\nThis might take a while...')
    start = time.time()

    fmts = ['.tar.bz2', '.tbz2', '.tar.gz', '.tgz', '.tar', '.tar.xz', '.txz', '.zip']

    for fname in os.listdir(src_dir):
        path = os.path.join(src_dir,fname)
        if os.path.splitext(fname)[1] in fmts:
            shutil.unpack_archive(path, dest_dir)

        elif os.path.isfile(path):
            shutil.copy(path, dest_dir)

        elif os.path.isdir(path):
            shutil.copytree(path,os.path.join(dest_dir,fname))

    end = time.time()
    print('Finished. Time taken: %.2fs.' %(end - start))

def extract_features(model, sample_count, generator, num_classes, batch_size = 32):
    """Extracts features and labels from a model.
       Use this for fast feature extraction.

    # Arguments
        model: The model to extract features from.
        sample_count: No. of expected samples.
        generator: Directory Iterator. Corresponds to 'ImageDataGenerator.flow_from_directory'.
    """
    feature_shape = (sample_count,) + ((model.layers[-1]).output_shape)[1:]
    features = np.zeros(shape = feature_shape)
    labels = np.zeros(shape=(feature_shape[0],num_classes))
    
    i = 0
    for inputs_batch, labels_batch in generator:        
        features_batch = model.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count: break

    return features, labels

def data_generator(directory, target_size, batch_size = 32, seed = 0, augment = False):
    """This is a wrapper around the 'flow_from_directory' method of the 'ImageDataGenerator' class
       with preset parameters for data augmentation. 

    # Arguments
        directory, target_size, batch_size, seed: Arguments for 'flow_from_directory' method.
        augment: If 'True', allows real-time data augmentation, else rescales the data.
    """
    datagen = ImageDataGenerator(rescale=1./255)
    
    if augment: datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest')


    return datagen.flow_from_directory(
                directory,
                target_size = target_size,
                batch_size = batch_size,
                seed = seed,
                class_mode = 'categorical')


def plot_history(history):
    """Plot the loss and accuracy curves for training and validation.

    # Arguments
        history: The 'History' object returned by 'fit' family of methods in Keras.

    """
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1,len(acc)+1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()