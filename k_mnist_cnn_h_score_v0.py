'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.

V0: 2018-9-18 22:40:55
'''

from __future__ import print_function
import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Lambda
from keras import backend as K
import numpy as np


def neg_hscore(x):
    """
    negative hscore calculation
    """
    f = x[0]
    g = x[1]
    f0 = f - K.mean(f, axis = 0)
    g0 = g - K.mean(g, axis = 0)
    corr = tf.reduce_mean(tf.reduce_sum(tf.multiply(f0, g0), 1))
    cov_f = K.dot(K.transpose(f0), f0) / K.cast(K.shape(f0)[0] - 1, dtype = 'float32')
    cov_g = K.dot(K.transpose(g0), g0) / K.cast(K.shape(g0)[0] - 1, dtype = 'float32')
    return - corr + tf.trace(K.dot(cov_f, cov_g)) / 2


batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# network f(x)
fdim = 128
gdim = fdim
input_x = Input(shape = input_shape)
conv1 = Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape)(input_x)
conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
pool = MaxPooling2D(pool_size=(2, 2))(conv2)
f = Dropout(0.25)(pool)
f = Flatten()(f)
f = Dense(fdim, activation='relu')(f)
f = Dropout(0.5)(f)

# network g(y)
input_y = Input(shape = (num_classes, ))
g = Dense(gdim)(input_y)

loss = Lambda(neg_hscore)([f, g])
model = Model(inputs = [input_x, input_y], outputs = loss)
model.compile(optimizer=keras.optimizers.Adadelta(), loss = lambda y_true,y_pred: y_pred)
model.fit([x_train, y_train], np.zeros([y_train.shape[0], 1]), batch_size = batch_size, epochs = epochs, validation_data=([x_test, y_test], np.zeros([y_test.shape[0], 1])))#validation_split = 0.2)
model_f = Model(inputs = input_x, outputs = f)
model_g = Model(inputs = input_y, outputs = g)
f_test = model_f.predict(x_test)
f_test = f_test - np.mean(f_test, axis = 0)
g_test = model_g.predict(np.eye(10))
g_test = g_test - np.mean(g_test, axis = 0)
py = np.mean(y_train, axis = 0)
pygx = py * (1 + np.matmul(f_test, g_test.T))
acc = np.mean(np.argmax(pygx, axis = 1) == np.argmax(y_test, axis = 1))
print(acc)


