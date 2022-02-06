## Implement multiple GAN models.

## TODO
### GAN
### WGAN

# ref:
## https://keras.io/examples/generative/dcgan_overriding_train_step/

# paper:
## GAN - https://arxiv.org/pdf/1701.07875.pdf 

import os 
import pickle as pkl
import tensorflow
from tensorflow import keras 
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Dense, Conv2D, UpSampling2D, Conv2DTranspose, Activation, 
                                            BatchNormalization, LeakyReLU, Flatten, Dropout, Reshape)
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam, RMSprop
import numpy as np 
import matplotlib.pyplot as plt 

## create a dataset from a folder, and rescale the images to [0, 1] range 
dataset = keras.preprocessing.image_dataset_from_directory(...)
dataset = dataset.map(lambda x: x / 255.0)


class GAN:
    def __init__(self, input_dim, discriminator_conv_filters: int, discriminator_conv_kernel_size: int, 
            discriminator_conv_strides: int, discriminator_batch_norm_momentum: float, discriminator_activation,
            discriminator_dropout_rate:float, discriminator_learning_rate:float,
            generator_init_dense_layer_size: int, generator_upsample, generator_conv_filters: int,
            generator_conv_kernel_size: int, generator_conv_strides: int, generator_batch_norm_momentum: float,
            generator_activation, generator_dropout_rate: float, generator_learning_rate: float,
            optimizer, z_dim):

        self.name = 'gan'
        self.input_dim = input_dim
        self.optimizer = optimizer
        self.z_dim = z_dim 

        ## discriminator
        self.discriminator_conv_filters = discriminator_conv_filters
        self.discriminator_conv_kernel_size = discriminator_conv_kernel_size
        self.discriminator_conv_strides = discriminator_conv_strides
        self.discriminator_batch_norm_momentum = discriminator_batch_norm_momentum
        self.discriminator_activation = discriminator_activation
        self.discriminator_dropout_rate = discriminator_dropout_rate
        self.discriminator_learning_rate = discriminator_learning_rate

        ## generator
        self.generator_init_dense_layer_size = generator_init_dense_layer_size
        self.generator_upsample = generator_upsample
        self.generator_conv_filters = generator_conv_filters 
        self.generator_conv_kernel_size = generator_conv_kernel_size 
        self.generator_conv_strides = generator_conv_strides
        self.generator_batch_norm_momentum = generator_batch_norm_momentum
        self.generator_activation = generator_activation
        self.generator_dropout_rate = generator_dropout_rate
        self.generator_learning_rate = generator_learning_rate

        self.n_layers_discriminator = len(discriminator_conv_filters)
        self.n_layers_generator = len(generator_conv_filters)

        self.weight_init = RandomNormal(mean=0.0, stddev=0.02)

        self.disc_losses = []
        self.gen_losses = []

        self.epoch = 0 

        self._build_discriminator()
        self._build_generator()

    def get_activation(self, activation):
        if activation == 'leaky_relu':
            layer = LeakyReLU(alpha=0.2)
        
        else:
            layer = Activation(activation)
        
        return layer

############# DISCRIMINATOR #############
        
    def _build_discriminator(self):
        discriminator_input = Input(shape=self.input_dim, name='discriminator_input')

        model = discriminator_input

        for i in range(self.n_layers_discriminator):
            model = Conv2D(
                    filters=self.discriminator_conv_kernel_size[i],
                    strides=self.discriminator_conv_strides[i],
                    padding='same',
                    name = 'discriminator_input_' + str(i),
                    kernel_initializer=self.weight_init
                )(model)
            
            if self.discriminator_batch_norm_momentum and i > 0:
                model = BatchNormalization(momentum=self.discriminator_batch_norm_momentum)(model)
            
            model = self.get_activation(self.discriminator_activation)(model)

            if self.discriminator_dropout_rate:
                model = Dropout(rate=self.discriminator_dropout_rate)(model)
        
        model = Flatten()(model)

        discriminator_output = Dense(1, activation='sigmoid', kernel_initializer=self.weight_init)(model)

        self.discriminator = Model(discriminator_input, discriminator_output)

        Model.summary()

############# GENERATOR #############
    
    def _build_generator(self):
        generator_input = Input(shape=(self.z_dim, ), name='generator_input')

        model = generator_input

        model = Dense(np.prod(self.generator_init_dense_layer_size), kernel_initializer=self.weight_init)(model)

        if self.generator_batch_norm_momentum:
            model = BatchNormalization(momentum=self.generator_batch_norm_momentum)(model)
        
        model = self.get_activation(self.generator_activation)(model)

        model = Reshape(self.generator_init_dense_layer_size)(model)

        if self.generator_dropout_rate:
            model = Dropout(rate=self.generator_dropout_rate)(model)
        
        for i in range(self.n_layers_generator):
            if self.generator_upsample[i] == 2:
                model = UpSampling2D()(model)

                model = Conv2D(
                    filters=self.generator_conv_filters[i],
                    kernel_size=self.generator_conv_kernel_size[i],
                    padding='same',
                    name='generator_conv_' + str(i),
                    kernel_initializer=self.weight_init
                )(model)
            
            else:
                model = Conv2DTranspose(
                    filters=self.generator_conv_filters[i],
                    kernel_size=self.generator_conv_kernel_size[i],
                    padding='same',
                    strides=self.generator_conv_strides[i],
                    name='generator_conv_' + str(i),
                    kernel_initializer= self.weight_init
                )(model)
            
            if i < self.n_layers_generator - 1:
                if self.generator_batch_norm_momentum:
                    model = BatchNormalization(momentum=self.generator_batch_norm_momentum)(model)
                
                model = self.get_activation(self.generator_activation)(model)
            
            else:
                model = Activation('tanh')(model)

        generator_output = model 

        self.generator = Model(generator_input, generator_output)

        Model.summary()
    
    def get_optim(self, learning_rate: float):
        if self.optimizer == 'adam':
            optim = Adam(learning_rate=learning_rate, beta_1=0.5)
        
        elif self.optimizer =='rmsprop':
            optim = RMSprop(learning_rate=learning_rate)
        
        else:
            optim = Adam(learning_rate=learning_rate)
        
        return optim
    
    def set_trainable(self, model, val):
        model.trainable = val 

        for l in model.layers:
            l.trainable = val 
    
    def _build_adversarial(self):
        self.discriminator.compile(
            optimizer = self.get_optim(self.discriminator_learning_rate),
            loss = 'binary_crossentropy',
            metrics = ['accuracy']
        )

        self.set_trainable(self.discriminator, False)

        model_input = Input(shape=(self.z_dim, ), name='model_input')
        model_output = self.discriminator(self.generator(model_input))

        self.model = Model(model_input, model_output)

        self.model.compile(
            optimizer=self.get_optim(self.generator_learning_rate),
            loss = 'binary_crossentropy',
            metrics=['accuracy']
        )

        self.set_trainable(self.discriminator, True)

############# TRAIN METHODS ############# 

    def train_discriminator(self, x_train, batch_size, using_generator):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        if using_generator:
            true_imgs = next(x_train)[0]

            if true_imgs.shape[0] != batch_size:
                true_imgs = next(x_train)[0]
        
        else:
            idx = np.random.ranint(0, x_train.shape[0], batch_size)
            true_imgs = x_train[idx]
        
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        generate_imgs = self.generator.predict(noise)

        disc_loss_real, disc_acc_real = self.discriminator.train_on_batch(true_imgs, valid)
        disc_loss_fake, disc_acc_fake = self.discriminator.train_on_batch(generate_imgs, fake)
        
        disc_loss = 0.5 * (disc_loss_real + disc_loss_fake)
        disc_acc = 0.5 * (disc_acc_real + disc_acc_fake)

        return [disc_loss, disc_loss_real, disc_loss_fake, disc_acc, disc_acc_real, disc_acc_fake]

    def train_generator(self, batch_size):
        valid = np.ones((batch_size, 1))
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))

        return self.model.train_on_batch(noise, valid)
    
    def train(self, x_train, batch_size, epochs, 
                run_folder, print_every_nth_batches: int=50, using_generator: bool=False):
        
        for epoch in range(self.epoch, self.epoch + epochs):
            disc = self.train_discriminator(x_train, batch_size, using_generator)
            gen = self.train_generator(batch_size)

            print(f'{epoch} [Discriminator loss: {disc[0]:.3f} (Real: {disc[1]:.3f}, Fake: {disc[2]:.3f}] \
                [Discriminator acc: ({disc[3]:.3f})({disc[4]:.3f}, {disc[5]:.3f}) \
                [Generator loss: {gen[0]:.3f}] [Generator acc: {gen[1]:.3f}]')
        
            self.disc_losses.append(disc)
            self.gen_losses.append(gen)

            if epoch % print_every_nth_batches == 0:
                self.sample_images(run_folder)
                self.model.save_weights(os.path.join(run_folder, f'weights/weights-{epoch}.h5'))
                self.model.save_weights(os.path.join(run_folder, f'weights/weights.h5'))
                self.save_model(run_folder)

            self.epoch += 1

    def sample_images(self, run_folder):
        rows, cols = 5, 5

        noise = np.random.normal(0, 1, (rows * cols, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = 0.5 * (gen_imgs + 1)
        gen_imgs = np.clip(gen_imgs, 0, 1)

        fig, axis = plt.subplot(rows, cols, figsize=(15, 15))
        
        counter = 0

        for i in range(rows):
            for j in range(cols):
                axis[i, j].imshow(np.squeeze(gen_imgs[counter, :, :]), cmap='gray')
                axis[i, j].axis('off')
                counter += 1
        
        fig.savefig(os.path.join(run_folder, f'images/sample_{self.epoch}.png'))
        plt.close()


    def save(self, folder):
        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pkl.dump([
                self.input_dim,
                self.discriminator_conv_filters,
                self.discriminator_conv_strides,
                self.discriminator_batch_norm_momentum,
                self.discriminator_activation,
                self.discriminator_dropout_rate,
                self.discriminator_learning_rate,
                
                self.generator_init_dense_layer_size,
                self.generator_upsample,
                self.generator_conv_filters,
                self.generator_conv_kernel_size,
                self.generator_conv_strides,
                self.generator_batch_norm_momentum,
                self.generator_activation,
                self.generator_dropout_rate,
                self.generator_learning_rate,

                self.optimizer,
                self.z_dim
            ], f)

    def save_model(self, run_folder):
        self.model.save(os.path.join(run_folder, 'model.h5'))
        self.discriminator.save(os.path.join(run_folder, 'discriminator.h5'))
        self.generator.save(os.path.join(run_folder, 'generator.h5'))

        pkl.dump(self.open(os.path.join(run_folder, 'obj.pkl'), 'wb'))

    def load_weights(self, filepath):
        self.model.load_weights(filepath)