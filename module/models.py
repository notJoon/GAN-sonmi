import argparse
from typing import Any

import torch
import torch.nn as nn

## some parsers for control hyperparam
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=150, help="number of epochs of training")
parser.add_argument('--z_dim', type=int, default=100, help='generator\'s input noise')
option = parser.parse_args()

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
    """ DCGAN Generator layer"""
    def __init__(self, img_channels=1, dim=8, z_dim=100) -> None:
        super(Generator, self).__init__()
        
        assert img_channels >= 1, f"img_channel must be 1(greyscale) or 3(RGB). got={img_channels}"
        assert dim >= 1, f"dim size cannot be zero or negative. the minimum recommend value is 8, got={dim}"
        assert z_dim >= 100, f"z_dim must greater or equal than 100. got={z_dim}"

        self.channels = img_channels
        self.z_dim = z_dim 

        def generator_block(self, in_filters:int, out_filters:int, first_block:bool) -> list:

            layers = []
            layers.append(nn.ConvTranspose2d(
                in_channels = in_filters,
                out_channels = out_filters,
                kernel_size = 4,
                stride = 1 if first_block else 2,
                padding = 0 if first_block else 1
            ))

            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout())

            return layers

        ### generate layers ###
        layers = []
        in_filters = self.z_dim

        for i, out_filters in enumerate([dim*8, dim*4, dim*2, dim]):
            layers.extend(generator_block(
                self,
                in_filters = in_filters,
                out_filters = out_filters,
                first_block = (i==0)
            ))
            in_filters = out_filters

        layers.append(nn.ConvTranspose2d(
            in_channels = in_filters,
            out_channels = self.channels,
            kernel_size = 4,
            stride = 2,
            padding = 1
        ))
        layers.append(nn.Tanh())

        self.models = nn.Sequential(*layers)

        print(self.models)

    def forward(self, x):
        return self.models(x)


class Discriminator(nn.Module):
    """ DCGAN Discriminator layer"""
    def __init__(self, img_channels=1, dim=8) -> None:
        super(Discriminator, self).__init__()

        assert img_channels >= 1, f"img_channel size must be 1(greyscale) or 3(RGB). got={img_channels}"

        self.channels = img_channels

        def discriminator_block(self, in_filters: int, out_filters: int) -> list:
            layers = []
            layers.append(nn.Conv2d(
                in_channels = in_filters, 
                out_channels = out_filters, 
                kernel_size = 4, 
                stride = 2, 
                padding = 1, 
                bias = False))

            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(0.5))

            return layers

        # discriminator layers
        filters: int = 64

        layers = []
        layers.append(nn.Conv2d(
            in_channels = self.channels, 
            out_channels = filters,
            kernel_size = 4,
            stride = 2,
            padding = 1))
        layers.append(nn.Sigmoid())
        
        input = filters
        for _, output in enumerate([dim, dim*2, dim*4, dim*8]):
            layers.extend(discriminator_block(
                self, 
                in_filters = input,
                out_filters = output
            ))
            
            input = output

        layers.append(nn.Conv2d(
            in_channels = output,
            out_channels = 1,
            kernel_size = 4,
            stride = 1,
            padding = 0
        ))
        layers.append(nn.Sigmoid())

        self.models = nn.Sequential(*layers)
        print(self.models)
    
    def forward(self, x):
        output = self.models(x)
        return output.squeeze()



### WGAN-GP Layer ###
class WGDiscriminator(nn.Module):
    """ WGAN-GP Discriminator layer """

    def __init__(self, img_channels=1, dim=32) -> None:
        super(WGDiscriminator, self).__init__()

        assert img_channels >= 1, f"img_channel size must be 1(greyscale) or 3(RGB). got={img_channels}"
        assert dim >= 1, f"dim size cannot be zero or negative. the minimum recommend value is 32, got={dim}"

        self.img_channels = img_channels

        def discriminator_block(self, in_filters:int, out_filters:int, 
                                first_block:bool=False) -> None:

            assert in_filters > 0, f"in_filters must greater than zero, got={in_filters}"
            assert out_filters > 0, f"out_filters must greater than zero, got={out_filters}"

            layers = []
            layers.append(nn.Conv2d(
                in_channels = in_filters,
                out_channels = out_filters,
                kernel_size = 4,
                stride = 2,
                padding = 1
            ))

            if first_block:
                layers.append(nn.InstanceNorm2d(out_filters))
            else:
                layers.append(nn.InstanceNorm2d(in_filters*2))
            
            layers.append(nn.LeakyReLU(0.2))

            return layers

        layers = []
        in_filters = self.img_channels

        for i, out_filters in enumerate([dim, dim*2, dim*4, dim*8]):
            layers.extend(discriminator_block(
                self,
                in_filters = in_filters,
                out_filters = out_filters,
                first_block = (i==0),
            ))

            in_filters = out_filters

        layers.append(nn.Conv2d(
                in_channels = out_filters, 
                out_channels = 1,
                kernel_size = 4,
                stride = 1,
                padding = 0
            ))
        
        self.models = nn.Sequential(*layers)
        
        print(self.models)

    def forward(self, x):
        return self.model(x).squeeze()


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)