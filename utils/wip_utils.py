'''
Work under Progress: Use this to and create custom utility functions.
'''

from keras.models import Sequential, Model
from keras.layers import Dense
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.optimizers import RMSprop
from tensorflow.python.ops import gradient_checker

model = Sequential()
model.add(Dense(10, activation = 'relu', input_shape = (10,)))
model.add(Dense(1,activation = 'sigmoid'))

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



train_X = np.ones(shape=(1000,10))
train_y = np.random.randint(2,size=(1000,1))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_X, train_y, epochs=10)

# test_X = np.random.rand(1000,10)
# y_pred = model.predict_classes(test_X).reshape(-1,)
# y_true = train_y.reshape(-1,)

# X = K.placeholder(ndim=1)
# Y = K.placeholder(ndim=1)
# Y_p = K.placeholder(ndim=1)

# print(model.output)

# loss = K.categorical_crossentropy(Y, Y_p)
# symbolic_gradient = K.function(inputs = (X, Y, Y_p), outputs = K.gradients(loss, X)) #function to call the gradient
# x = np.ones(shape=(10,1))
# print(symbolic_gradient([x, y_true, y_pred]))
# # updates = optimizer.get_updates(model.trainable_weights, model.constraints, cost))
# loss = model.layers[-1].activation(X)

def symbolic_gradients(model, input, output):
    grads = K.gradients(model.total_loss, model.trainable_weights)
    inputs = model.model._feed_inputs + model.model._feed_targets + model.model._feed_sample_weights
    fn = K.function(inputs, grads)

    return fn([input, output, np.ones(len(output))])


print(symbolic_gradients(model, train_X, train_y))

print(numerical_gradients(model))

# x = np.random.random((128,)).reshape((-1, 1))
# y = 2 * x
# model = Sequential(layers=[Dense(2, input_shape=(1,)),
#                             Dense(1)])
# model.compile(loss='mse', optimizer='rmsprop')
# get_gradient = get_gradient_norm_func(model)
# history = model.fit(x, y, epochs=1)
# print(get_gradient([x, y, np.ones(len(y))]))


# import numpy as np
# from keras import objectives
# from keras import backend as K

# _EPSILON = K.epsilon()

# def _loss_tensor(y_true, y_pred):
#     y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
#     out = -(y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred))
#     return K.mean(out, axis=-1)

# def _loss_np(y_true, y_pred):
#     y_pred = np.clip(y_pred, _EPSILON, 1.0-_EPSILON)
#     out = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
#     return np.mean(out, axis=-1)

# def check_loss(_shape):
#     if _shape == '2d':
#         shape = (6, 7)
#     elif _shape == '3d':
#         shape = (5, 6, 7)
#     elif _shape == '4d':
#         shape = (8, 5, 6, 7)
#     elif _shape == '5d':
#         shape = (9, 8, 5, 6, 7)

#     y_a = np.random.random(shape)
#     y_b = np.random.random(shape)

#     out1 = K.eval(_loss_tensor(K.variable(y_a), K.variable(y_b)))
#     out2 = _loss_np(y_a, y_b)

#     assert out1.shape == out2.shape
#     assert out1.shape == shape[:-1]
#     print np.linalg.norm(out1)
#     print np.linalg.norm(out2)
#     print np.linalg.norm(out1-out2)


# def test_loss():
#     shape_list = ['2d', '3d', '4d', '5d']
#     for _shape in shape_list:
#         check_loss(_shape)
#         print '======================'

# if __name__ == '__main__':
#     test_loss()`

# sprint(K.eval(model.total_loss))