import os
import zipfile
import shutil
import time
from keras import models

def unzip_dataset(src_dir, dest_dir=None, cleanup=False):
    '''
    Unzips zipped files in 'src_dir' to 'dest_dir'.
    If 'cleanup' = True, will delete the zip files.
    '''
    # use this for debugging
    #print(os.getcwd())
    
    if not dest_dir:
        dest_dir = src_dir

    print("Extracting from '{}'' to '{}'.".format(src_dir,dest_dir))
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

'''def load_data(m=None,path=None):
   
    The function expects by default 32x32 size images and returns a tuple.
    Arguments: m - # of examples, path - path/to/images 
    Returns: (data, labels)
    
    shape = (m, n_H, n_W,n_C)
    n_H - image height
    n_W - image width
    n_C - # of channels
      
    if not path: 
        path = 'data/imgs/'
    if not m:
        m = len(os.listdir(path))
    else: 
        assert type(m) == int
    
    shape = (m,32,32,1)
    m_y0 = len([filename for filename in os.listdir(path) if filename.startswith('no-tick')])
    m_y1 = len([filename for filename in os.listdir(path) if filename.startswith('tick')])
            
    X = np.empty(shape, dtype='uint8')
    y0, y1 = np.zeros(shape = (m_y0,)), np.ones(shape = (m_y1,))
    y = np.concatenate((y0,y1))
    
    
    for i, img in enumerate(os.listdir(path)): 
        X[i] = cv2.imread(os.path.join(path, img),0).reshape(shape[1:])
        
    #shuffle(X,y)
      
    return X, y'''