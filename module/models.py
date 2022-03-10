import argparse
from typing import Any

import torch
import torch.nn as nn

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
### DCGAN Layer ###
class Generator(nn.Module):

    def __init__(self, generator_channels=3, conv_trans_strides=1, 
                generator_filters=64, kernel_size=4, conv_trans_paddings=0, z_dim=100) -> None:
        super(Generator, self).__init__()

        self.generator_channels = generator_channels
        self.conv_trans_strides = conv_trans_strides
        self.generator_filters = generator_filters
        self.kernel_size = kernel_size
        self.conv_trans_paddings = conv_trans_paddings
        self.z_dim = z_dim 



        def generator_block(
            self, in_filters: int, 
            out_filters: int, 
            first_block = False) -> list:
            layers = []

            layers.append(nn.ConvTranspose2d(
                in_channels=in_filters,
            ))

            layers.append(nn.BatchNorm2d(self.generator_filters))
            layers.append(nn.ReLU(True))
            layers.append(nn.Dropout())

            return layers

        ### generate layers ###
        layers = []
        input = self.z_dim

        for i, output in enumerate([512, 256, 128, 64]):
            layers.extend(generator_block(
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

    def forward(self, x):
        return self.models(x)



class Discriminator(nn.Module):

    def __init__(
        self, discriminator_channels=3, 
        discriminator_kernels=4, discriminator_strides=2, 
        discriminator_paddings=1) -> None:

        super(Discriminator, self).__init__()
        self.discriminator_channels = discriminator_channels
        self.discriminator_kernels = discriminator_kernels
        self.discriminator_strides = discriminator_strides
        self.discriminator_paddings = discriminator_paddings
        self.leaky_relu_rate = 0.2
        self.dropout_rate = 0.5

        def discriminator_block(self, in_filters: int, out_filters: int) -> list:
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
            layers.extend(discriminator_block(
                self, 
                in_filters = input,
                out_filters = output))
            
            output = input 
        
        self.models = nn.Sequential(*layers)
        print(self.models)
    
    def forward(self, x):
        output = self.models(x)
        return output.squeeze()

### WGAN-GP Layer ###
class WGDiscriminator(nn.Module):
    def __init__(
        self, conv_kernel_size: int = 4, conv_strides: int = 2, 
        conv_paddings: int = 1, leaky_rate: float = 0.2, dim: int = 32) -> None:
        super(WGDiscriminator, self).__init__()
        
        self.input = 1 
        self.conv_kernel_size = conv_kernel_size
        self.conv_strides = conv_strides
        self.conv_paddings = conv_paddings
        self.leaky_rate = leaky_rate
        self.dim = dim

        """ WGAN-GP layer example 
        Sequential(
            (0): Conv2d(1, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (2): LeakyReLU(negative_slope=0.2)
            (3): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            (4): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (5): LeakyReLU(negative_slope=0.2)
            (6): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            (7): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (8): LeakyReLU(negative_slope=0.2)
            (9): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            (10): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (11): LeakyReLU(negative_slope=0.2)
            (12): Conv2d(256, 1, kernel_size=(4, 4), stride=(1, 1))
            )
        """

        def discriminator_block(self, in_filters: int, out_filters: int, first_block: bool = False) -> None:
            layers = []
            layers.append(nn.Conv2d(
                in_channels = in_filters,
                out_channels = out_filters,
                kernel_size = self.conv_kernel_size,
                stride = self.conv_strides,
                padding = self.conv_paddings
            ))

            if first_block:
                layers.append(nn.InstanceNorm2d(out_filters))
            else:
                layers.append(nn.InstanceNorm2d(in_filters * 2))
            
            layers.append(nn.LeakyReLU(self.leaky_rate))

            return layers

        layers = []
        in_filters = self.input

        for i, out_filters in enumerate([32, 64, 128, 256]):
            layers.extend(discriminator_block(
                self,
                in_filters = in_filters,
                out_filters = out_filters,
                first_block = (i==0),
            ))

            in_filters = out_filters

        layers.append(nn.Conv2d(
                out_filters, 
                out_channels = 1,
                kernel_size = 4,
                stride = 1,
                padding = 0
            ))
        
        self.models = nn.Sequential(*layers)
        print(self.models)

    def forward(self, x):
        return self.model(x).squeeze()