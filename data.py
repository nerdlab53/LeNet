import tensorflow as tf
from tensorflow.keras import datasets, models, layers, losses
import matplotlib.pyplot as plt


# loading the MNIST dataset for classification
# As the original LeNet takes in 32 x 32 images :  
# - the 28 x 28 images are padded with zeros
# - 8-bit pixel values are scaled between 0-1
# - also a dummy dimension is added in the end

def load_data():
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
    X_train = tf.pad(X_train, [[0, 0], [2, 2], [2, 2]])/255
    X_test = tf.pad(X_test, [[0, 0], [2, 2], [2, 2]])/255
    X_train = tf.expand_dims(X_train, axis=3)
    X_test = tf.expand_dims(X_test, axis=3)
    X_val = X_train[-2000:, :, :, :]
    y_val = y_train[-2000:]
    X_train = X_train[:-2000,:,:,:] 
    y_train = y_train[:-2000]
    return X_train, y_train, X_val, y_val, X_test, y_test