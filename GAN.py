import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Flatten
from tensorflow.keras.optimizers import Adam


(X_train, _), _ = mnist.load_data()
X_train = (X_train - 127.5) / 127.5  
X_train = X_train.reshape(-1, 784)

# Optimizer
opt = Adam(0.0002, 0.5)

def build_generator():
    model = Sequential([
        Dense(128, input_dim=100),
        LeakyReLU(0.2),
        Dense(784, activation='tanh'), 
    ])
    return model


def build_discriminator():
    model = Sequential([
        Dense(128, input_shape=(784,)),
        LeakyReLU(0.2),
        Dense(1, activation='sigmoid')  
    ])
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

