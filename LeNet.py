import tensorflow as tf
from tensorflow.keras import datasets, models, layers, losses
import matplotlib.pyplot as plt

# Defining the architecture of the net
# The LeNet architecture is as follows : 
# LeNet Architecture : 
#- Input : 28x28
#- 6 Conv2D Layers with 28x28
#- 6 Pooling Layers with 14x14
#- 16 Conv2D Layers with 10x10
#- 16 Pooling Layers with 5x5
#- Fully Connected Dense Layer 120
#- Fully Connected Dense Layer 84
#- Fully Connected Dense Layer 10

def model():
    model = models.Sequential()
    model.add(layers.Conv2D(6, 5, activation='tanh', input_shape=(32, 32, 1)))
    model.add(layers.AveragePooling2D(2))
    model.add(layers.Conv2D(16, 5, activation='tanh'))
    model.add(layers.AveragePooling2D(2))
    model.add(layers.Activation('sigmoid'))
    model.add(layers.Conv2D(120, 5, activation='tanh'))
    model.add(layers.Flatten())
    model.add(layers.Dense(84, activation='tanh'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

def plot_model(history):
    fig, axs = plt.subplots(2, 1, figsize=(15, 15))
    axs[0].plot(history.history['loss'])
    axs[0].plot(history.history['val_loss'])
    axs[0].title.set_text('Training Loss vs Validation Loss')
    axs[0].legend(['Train', 'Val'])
    axs[1].plot(history.history['acc'])
    axs[1].plot(history.history['val_acc'])
    axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
    axs[1].legend(['Accuracy', 'Validation Accuracy'])