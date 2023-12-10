from LeNet import model, plot_model
from data import load_data
import tensorflow as tf
from tensorflow.keras import models, datasets, losses, layers
import matplotlib.pyplot as plt


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    LeNet = model()
    LeNet.summary()
    LeNet.compile(optimizer='adam',
                  loss=losses.sparse_categorical_crossentropy,
                  metrics=['acc'])
    history = LeNet.fit(X_train, y_train, batch_size=64, epochs=15, validation_data=(X_val, y_val))
    # Plotting the model
    plot_model(history)


if __name__ == "__main__":
    main()