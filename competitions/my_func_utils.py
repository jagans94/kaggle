import os
from keras import models

def load_model(path=None):
    '''
    Loads a model from the default directory `../saved_models`.
    '''
    model_dir = '../saved_models'
    model_name = input("Enter model name.(Example: fashion_mnist_v1.h5)\n") 
    
    path = os.path.join(model_dir,model_name)
    assert os.path.isfile(path)
    
    return models.load_model(path)

def save_model(model):
    '''
    Saves the given model to the default directory `../saved_models`.
    '''
    model_dir = '../saved_models'
    model_name = input("Enter model name. (Example: fashion_mnist_v1.h5)\n")
    
    path = os.path.join(model_dir,model_name)
    assert os.path.isfile(path)

    model.save(path)