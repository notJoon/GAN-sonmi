import argparse
from typing import Any

import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim

## TODO finishing train model 

## some parsers for control hyperparam
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=150, help="number of epochs of training")
parser.add_argument('--dropout_rate', type=float, default=0.5, help="set droput rate")
parser.add_argument('--leakyrelu_rate', type=float, default=0.2, help="set LeakyReLU learning rate")
parser.add_argument('--z_dim', type=int, default=100, help='generator\'s input noise')
option = parser.parse_args()

print(option)

""" Architectue

    ### GAN ###

        | real(x) | ───────────────────────────────x────────────────────────> | discriminator | ──> | output(y_hat) |
                                                   │
                                             | fake(x_hat) |
                                                   │
        | noise | ──────────>| generator |─────────┘
            |
            *
  //  diverse representation  //
  //  and also NOT real image //


    ### discriminator ###
         * learns to distinguish real from fake
                                                                        ┌─────────── update(theta_d) ───────────┐
                                                                        v                                       │
        | noise | ──> | generator | ──> | features(x_hat) | ─x─> | discriminator | ──> | output(y_hat) | ──> | cost |
                                                             │                                                  │
                                                             │                                                  *
                                                        | real(x) |                                 // Binary cross-entropy  //
                                                                                                    // with labels real/fake //

        ### generator ###
         * learns to make fakes that look real
         * take any random noise and produce a realistic image

                           ┌─────────────────────────────── update(theta_g) ───────────────────────────────────┐
                           V                                                                                   │                                                                 
        | noise | ──> | generator | ──> | features(x_hat) | ──> | discriminator | ──> | output(y_hat) | ──> | cost |
                                                │
                                                *
                                        //  only takes   //
                                        // fake examples //

       * they learn the competition with each other                                                         
       * the two models should always be at a similar skill level

"""
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class DCGenerator(nn.Module):

    def __init__(self, generator_channels=3, conv_trans_strides=1, 
                generator_filters=64, kernel_size=4, conv_trans_paddings=0, z_dim=100) -> None:
        super(DCGenerator, self).__init__()

        self.generator_channels = generator_channels
        self.conv_trans_strides = conv_trans_strides
        self.generator_filters = generator_filters
        self.kernel_size = kernel_size
        self.conv_trans_paddings = conv_trans_paddings
        self.z_dim = z_dim 



        def dc_generator_block(
            self, in_filters: int, 
            out_filters: int, 
            first_block = False) -> list:
            layers = []

            layers.append(nn.ConvTranspose2d(
                in_channels=in_filters,
                out_channels=out_filters,
                kernel_size=self.kernel_size,
                stride=self.conv_trans_strides,
                padding=self.conv_trans_paddings
            ))

            layers.append(nn.BatchNorm2d(self.generator_filters))
            layers.append(nn.ReLU(True))
            layers.append(nn.Dropout())

            return layers

        ### generate layers ###
        layers = []
        input = self.z_dim

        for i, output in enumerate([512, 256, 128, 64]):
            layers.extend(dc_generator_block(
                self,
                in_filters = input,
                out_filters = output,
                first_block = (i == 0)
            ))

            input = output

        layers.append(nn.ConvTranspose2d(
            in_channels = input,
            out_channels = self.generator_channels,
            kernel_size = self.kernel_size,
            stride = 2,
            padding = 1
        ))
        layers.append(nn.Tanh())

        self.models = nn.Sequential(*layers)

        print(self.models)

    def forward(self, input: Any) -> Any:
        output = self.models(input)
        return output



class DCDiscriminator(nn.Module):

    def __init__(
        self, discriminator_channels=3, 
        discriminator_kernels=4, discriminator_strides=2, 
        discriminator_paddings=1) -> None:

        super(DCDiscriminator, self).__init__()
        self.discriminator_channels = discriminator_channels
        self.discriminator_kernels = discriminator_kernels
        self.discriminator_strides = discriminator_strides
        self.discriminator_paddings = discriminator_paddings
        self.leaky_relu_rate = option.leakyrelu_rate
        self.dropout_rate = option.dropout_rate

        def dc_discriminator_block(self, in_filters: int, out_filters: int) -> list:
            layers = []
            layers.append(nn.Conv2d(
                in_channels = in_filters, 
                out_channels = out_filters, 
                kernel_size = self.discriminator_kernels, 
                stride = self.discriminator_strides, 
                padding = self.discriminator_paddings, 
                bias = False))

            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(self.leaky_relu_rate, inplace=True))
            layers.append(nn.Dropout(self.dropout_rate))

            return layers

        # generate discriminator's layers
        CONV_INPUT: int = 64

        layers = []
        layers.append(nn.Conv2d(
            in_channels = self.discriminator_channels, 
            out_channels = CONV_INPUT,
            kernel_size = self.discriminator_kernels,
            stride = self.discriminator_strides,
            padding = self.discriminator_paddings))
        layers.append(nn.Sigmoid())
        
        input = CONV_INPUT
        for _, output in enumerate([64, 128, 256, 512]):
            layers.extend(dc_discriminator_block(
                self, 
                in_filters = input,
                out_filters = output))
            
            output = input 
        
        self.models = nn.Sequential(*layers)
        print(self.models)

def train_discriminator(batch_size: int, z_dim: int = 100):
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))


### RUN TEST ### 
if __name__ == "__main__":
    ...
