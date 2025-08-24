import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Flatten
from tensorflow.keras.optimizers import Adam


(X_train, _), _ = mnist.load_data()
X_train = (X_train - 127.5) / 127.5  
X_train = X_train.reshape(-1, 784)

# Optimizer
opt = Adam(0.0002, 0.6)

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

generator = build_generator()
discriminator = build_discriminator()

noise = np.random.normal(0, 1, (128, 100))          
fake_images = generator.predict(noise)             

real_images = X_train[np.random.randint(0, X_train.shape[0], 128)]  

real_labels = np.ones((128, 1))
fake_labels = np.zeros((128, 1))


d_loss_real = discriminator.train_on_batch(real_images, real_labels)
d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)

print("Discriminator accuracy on real:", d_loss_real[1]*100)
print("Discriminator accuracy on fake:", d_loss_fake[1]*100)