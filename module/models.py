import argparse
from typing import Any

import torch
import torch.nn as nn
from save_load import SaveLoadData

def check_dim(dim: int, mode='normal') -> str:
    modes = ['normal', 'z']
    
    if mode == modes[0]:
        assert dim >= 1, f"dim size cannot be zero or negative. got={dim}"
    else:
        assert dim > 100, f"z_dim must greater or equal than 100. got={dim}"


def check_channels(img_channels: int) -> str:
    assert img_channels >= 1, f"img_channel must be 1(greyscale) or 3(RGB). got={img_channels}"

### DCGAN Layer ###
class Generator(nn.Module):
    """ DCGAN Generator layer"""
    __constants__ = ['dim', 'img_channels', 'z_dim']

    def __init__(self, dim: int, img_channels: int, z_dim=100) -> None:
        super(Generator, self).__init__()
    
        check_dim(dim)
        check_dim(z_dim, mode='z')
        check_channels(img_channels)

        self.channels = img_channels
        self.z_dim = z_dim 


        
        def _build_generator_block(self, in_filters:int, out_filters:int, first_block:bool) -> list:

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
            layers.extend(_build_generator_block(
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

        # print(self.models)

    def _forward(self, x):
        return self.models(x)
    
    def save(self):
        SaveLoadData.save_model(self.models)


class Discriminator(nn.Module):
    __constants__ = ['dim, img_channels, filters']

    def __init__(self, dim: int, img_channels: int, filters=64) -> None:
        super(Discriminator, self).__init__()

        check_dim(dim)
        check_channels(img_channels)

        self.channels = img_channels

        def _build_discriminator_block(self, in_filters: int, out_filters: int) -> list:
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

        # discriminator layers    # img_size
        layers = []
        layers.append(nn.Conv2d(
            in_channels = self.channels, 
            out_channels = filters,
            kernel_size = 4,
            stride = 2,
            padding = 1))
        layers.append(nn.Sigmoid())
        
        input = filters
        for _, output in enumerate([dim*2, dim*4, dim*8]):
            layers.extend(_build_discriminator_block(
                self, 
                in_filters = input,
                out_filters = output
            ))
            
            input = output

        layers.append(nn.Conv2d(
            in_channels = output,
            out_channels = 1,
            kernel_size = 4,
            stride = 2,
            padding = 0
        ))
        layers.append(nn.Sigmoid())

        self.models = nn.Sequential(*layers)
        #print(self.models)
    
    def _forward(self, x):
        models = self.models(x)
        return models.squeeze()
    
    def save(self):
        SaveLoadData.save_model(self.models)

#####################################



class ConditionalGenerator(nn.Module):    
    __constants__ = ['dim', ' num_class', 'img_size', 'embed_size', 'img_channels', 'z_dim']

    def __init__(
        self, 
        dim: int, 
        num_class: int,
        img_size: int,
        embed_size: int,
        img_channels: int, 
        z_dim=100) -> None:

        super(ConditionalGenerator, self).__init__()
        check_channels(img_channels)
        check_dim(dim)
        check_dim(z_dim, mode='z')

        self.img_size = img_size
        self.channels = img_channels
        self.z_dim = z_dim 


        def _build_generator_block(self, in_filters:int, out_filters:int, first_block:bool) -> list:

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
        in_filters = self.z_dim+embed_size

        for i, out_filters in enumerate([dim*8, dim*4, dim*2, dim]):
            layers.extend(_build_generator_block(
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
        self.embedding = nn.Embedding(num_class, img_size)

        # print(self.models)

    def _forward(self, x, labels):
        ## latent vector: N x noise_dim x 1 x1
        embedding = self.embedding(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        return self.models(x)
    
    def save(self):
        SaveLoadData.save_model(self.models)

class ConditionalDiscriminator(nn.Module):
    __constants__ = ['dim', 'img_channels', 'num_classes', 'img_size']

    def __init__(
        self, 
        dim: int, 
        img_channels: int, 
        num_classes: int, 
        img_size: int) -> None:
        super(ConditionalDiscriminator, self).__init__()

        check_dim(dim)
        check_channels(img_channels)

        self.channels = img_channels
        self.img_size = img_size

        def _build_discriminator_block(self, in_filters: int, out_filters: int) -> list:
            layers = []
            layers.append(nn.Conv2d(
                in_channels = in_filters, 
                out_channels = out_filters, 
                kernel_size = 4, 
                stride = 2, 
                padding = 1, 
                bias = False
            ))

            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(0.5))
            return layers

        # discriminator layers
        filters: int = 64       # img_size

        layers = []
        layers.append(nn.Conv2d(
            in_channels = self.channels+1, 
            out_channels = filters,
            kernel_size = 4,
            stride = 2,
            padding = 1))
        layers.append(nn.LeakyReLU(0.2))
        
        input = filters
        for _, output in enumerate([dim*2, dim*4, dim*8]):
            layers.extend(_build_discriminator_block(
                self, 
                in_filters = input,
                out_filters = output
            ))
            
            input = output

        layers.append(nn.Conv2d(
            in_channels = output,
            out_channels = 1,
            kernel_size = 4,
            stride = 2,
            padding = 0
        ))
        layers.append(nn.Sigmoid())

        self.models = nn.Sequential(*layers)
        self.embedding = nn.Embedding(num_classes, img_size*img_size)
        #print(self.models)
    
    def _forward(self, x, labels):
        embedding = self.embedding(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1) # N x C x img_size(H) x img_size(W)
        return self.models(x)
    
    def save(self):
        SaveLoadData.save_model(self.models)



### WGAN-GP Layer ###
class Critic(nn.Module):
    """ WGAN-GP Discriminator layer """
    __constants__ = ['dim', 'img_channels']

    def __init__(self, dim: int, img_channels: int) -> None:
        super(Critic, self).__init__()

        check_channels(img_channels)
        check_dim(dim)

        self.img_channels = img_channels

        def _build_discriminator_block(
            self, 
            in_filters: int, 
            out_filters: int, 
            first_block: bool) -> None:

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
                layers.append(nn.InstanceNorm2d(out_filters, affine=True))
            else:
                layers.append(nn.InstanceNorm2d(in_filters*2, affine=True))
            
            layers.append(nn.LeakyReLU(0.2))

            return layers

        layers = []
        in_filters = self.img_channels

        for i, out_filters in enumerate([dim, dim*2, dim*4, dim*8]):
            layers.extend(_build_discriminator_block(
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
        
        #print(self.models)

    def _forward(self, x):
        return self.model(x).squeeze()
    
    def save(self):
        SaveLoadData.save_model(self.models)