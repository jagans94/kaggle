from keras.models import Sequential, Model
from keras.layers import Dense
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.optimizers import RMSprop

model = Sequential()
model.add(Dense(10, activation = 'relu', input_shape = (10,)))
model.add(Dense(1,activation = 'sigmoid'))

layers_activations = [layer.activation for layer in model.layers]

# p_b4 = np.array(model.get_weights())
# print(p_b4[::2])
# for param in p_b4:
#     shape = param.shape
#     X = K.placeholder(shape=shape) #specify the right placeholder
#     loss = model.layers[-1].activation(X)
#     fn = K.function([X], K.gradients(loss, [X]) ) #function to call the gradient
#     print('\n')
#     grad = fn([param])
#     print(grad)
#     print('\n')


# train_X = np.random.rand(1000,100)
train_X = np.ones(shape=(1000,10))
train_y = np.random.randint(2,size=(1000,))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(train_X, train_y, epochs=10)

p_af = np.array(model.get_weights())
# print(p_af[::2])

def symbolic_gradient(model):
    """Returns symbolic gradient for
    
    # Arguments
        model:
    """



for param in p_af:
    shape = param.shape
    X = K.placeholder(shape=shape) 
    h = (1e-4,)
    loss = model.layers[-1].activation
    symbolic_gradient = K.function([X], K.gradients(loss(X), [X])) #function to call the gradient
    numerical_gradient = K.function([X], [(loss(X+h) - loss(X-h))/(2)])
    print('\n')
    g1 = symbolic_gradient([param])
    g2 = numerical_gradient([param])
    print(g1)
    print(g2)
    print('\n')
    break

# print(p_af[::2])
# print('\n\n')
# print(p_af - p_b4)

# print(model.get_config())