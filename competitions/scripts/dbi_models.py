"""
This python script contains models and other utility functions for 
the dog_breed_classification.ipynb file.
"""
import os
import numpy as np
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16

def basic(input_shape, num_classes, dropout = False):
    """Returns a basic CNN model.
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    if dropout: model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

def vgg16_conv_base(input_shape):
    """Returns a VGG16 CNN base. 
    """
    return VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

def conv_top(input_shape, num_classes):
    """Returns a CNN base model.
       Similar to the 'top' part of the basic model. 
    """
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=input_shape))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

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
    """Generates minibatches of image data with real-time data augmentation.
       This is a wrapper around the 'ImageDataGenerator' class. 

    # Arguments
        directory, target_size, batch_size, seed: Arguments for 'flow_from_directory' method.
        augment: If 'True', allows real-time data augmentation, else rescales the data.
    """
    datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest')

    if augment: datagen = ImageDataGenerator(rescale=1./255)

    return datagen.flow_from_directory(
                directory,
                target_size = target_size,
                batch_size = batch_size,
                seed = seed,
                class_mode = 'categorical')