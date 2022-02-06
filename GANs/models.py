## Implement multiple GAN models.

## TODO
### GAN
### WGAN

# ref:
## https://keras.io/examples/generative/dcgan_overriding_train_step/

# paper:
## GAN - https://arxiv.org/pdf/1701.07875.pdf 



import tensorflow
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Dense, Conv2D, Conv2DTranspose, Activation, 
                                            LeakyReLU, Flatten, Dropout, Reshape)
import numpy as np 
import matplotlib.pyplot as plt 

## create a dataset from a folder, and rescale the images to [0, 1] range 
dataset = keras.preprocessing.image_dataset_from_directory(...)
dataset = dataset.map(lambda x: x / 255.0)


class NormalGAN:
    def generator(self, latent_dim:int = 128):
        generator = Sequential([
            Input(shape=(latent_dim, )),
            Dense(8 * 8 * 128),
            Reshape((8, 8, 128)),

            Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'),
            LeakyReLU(alpha=0.2),

            Conv2DTranspose(256, kernel_size=4, strides=2, padding='same'),
            LeakyReLU(alpha=0.2),

            Conv2DTranspose(512, kernel_size=4, strides=2, padding='same'),
            LeakyReLU(alpha=0.2),

            Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')],
            name='generator')

        generator.summary()
    
    def discriminator(self):
        # it maps a 80x80 image to a binary classification score
        discriminator = Sequential([
            Input(shape=(64, 64, 1)),

            Conv2D(64, kernel_size=4, strides=2, padding='same'),
            LeakyReLU(alpha=0.2),

            Conv2D(128, kernel_size=4, strides=2, padding='same'),
            LeakyReLU(alpha=0.2), 

            Conv2D(128, kernel_size=4, strides=2, padding='same'),
            LeakyReLU(alpha=0.2),

            Flatten(),
            Dropout(rate=0.2),

            Dense(1, activation='sigmoid')],
        name = 'discriminator')
    discriminator.summary()

        
