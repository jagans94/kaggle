# general imports
import time
import os, shutil
import zipfile

import numpy as np

# specific imports
from keras import models
from sklearn.model_selection import train_test_split

def bin_dataset(directory, mapping, labels = [], validation_split = 0.0, random_state = 0):
    """Bins the dataset (present in the mapping representation) into respective classes.
       Use this to bin the entire dataset, or just a subset of it.
       Complements the 'ImageDataGenerator' class in Keras.

    # Arguments
        directory: Source directory.
        mapping: Numpy array of shape (N,2). First column should contain filenames/id and the second column 
                 corresponding ground truth labels. 
        [labels]: Class labels. Needs a minimum of 2 classes. If 'empty', inferred from the mapping. 
                  You can even pass a subset of labels, if required.
        [validation_split]: A float in the half-interval range [0,1). Used to split the 'dataset' into training and validation sets.
        [random_state] : Random seed for train/val split.

    """
    if labels == []: labels = np.unique(mapping[:,1])
    # no. of files in directory
    tnof = len([fname for fname in os.listdir(directory) if os.path.isfile(os.path.join(directory, fname))])

    # Checks
    assert len(dict(mapping)) == len(mapping), "The mapping is not hashable!!"
    assert os.path.isdir(directory), "The directory doesnot exist!!"
    assert mapping.shape == mapping.reshape(-1,2).shape, "The mapping provided is incorrect. Check the doc string for more info."
    assert labels.size > 1, "Too few labels!!"
    assert validation_split in np.arange(0.0,1.0,1e-3), "The value provided for 'validation_split' arg is invalid. Check the doc string for more info."
    assert tnof > 0, "There are no files in the directory."
    print('Tests passed.')
    print('Total no. of files: ', )
    print('Selected no. of files: ', len(mapping))

    # Helper functions
    def create_labels_folder(directory, labels):
        for label in labels:
            path = os.path.join(directory,label)
            if not os.path.exists(path):
                os.makedirs(path)
    
    def move_files_into_labels_folders(src_dir, dst_dir, mapping):
        for fname, label in mapping.items():
            fname += '.jpg'
            src = os.path.join(src_dir,fname)
            dst = os.path.join(dst_dir, os.path.join(label,fname))
            if os.path.isfile(src): shutil.move(src,dst)

    # Split and bin the files into training and validation sets.
    if validation_split > 0.0:
        # Computing the split
        train_path = os.path.join(directory,'train')
        val_path = os.path.join(directory, 'val')

        X, y = mapping[:,0], mapping[:,1]
        X_train, X_val, y_train, y_val = train_test_split(  
                                                        X, y,
                                                        test_size = validation_split, 
                                                        random_state=random_state, 
                                                        stratify = y,
                                                        )
        # Need to reshape the numpy arrays to use them.
        X_train = X_train.reshape(-1,1)
        y_train = y_train.reshape(-1,1)
        X_val = X_val.reshape(-1,1)
        y_val = y_val.reshape(-1,1)

        train_mapping = np.concatenate([X_train,y_train], axis = 1)
        val_mapping = np.concatenate((X_val,y_val), axis = 1)

        # Binning the files
        create_labels_folder(train_path, labels)
        create_labels_folder(val_path, labels)
        move_files_into_labels_folders(directory, train_path, dict(train_mapping))
        move_files_into_labels_folders(directory, val_path, dict(val_mapping))
    
    else:
        # Binning the files
        create_labels_folder(directory, labels)
        move_files_into_labels_folders(directory, directory, dict(mapping))  

def unzip_dataset(src_dir, dest_dir = None, cleanup = False):
    """Unzips zipped files in the 'src_dir' to 'dest_dir'.

    # Arguments
        src_dir: Source directory, i.e. where the zip files are.
        dest_dir: Destination directory.
        cleanup: Flag. Set this if you want to remove the zip files after extracting.

    """
    if not dest_dir:
        dest_dir = src_dir
    
   # Checks
    assert os.path.isdir(src_dir), "The source directory doesnot exist!!"
    assert os.path.isdir(dest_dir), "The destination directory doesnot exist!!"
    print('Tests passed.')

    print("Extracting from '{}' to '{}'.".format(src_dir,dest_dir))
    print('This might take some time...')
    start = time.time()

    pathnames = [os.path.join(src_dir,filename) for filename in os.listdir(src_dir) if filename.endswith('.zip')]
    for path in pathnames:
        with zipfile.ZipFile(path,"r") as zip_ref:    
            zip_ref.extractall(dest_dir)
    
    if cleanup:
        for path in pathnames: os.remove(path)

    end = time.time()
    print('Finished. Time taken: %.2fs.' %(end - start))