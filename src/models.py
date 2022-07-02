from typing import Any

import torch
import torch.nn as nn

def check_dim(dim: int, mode='normal') -> str:
    modes = ['normal', 'z']
    
    if mode == modes[0]:
        assert dim >= 1, f"dim size cannot be zero or negative. got={dim}"
    else:
        assert dim > 100, f"z_dim must greater or equal than 100. got={dim}"


def check_channels(img_channels: int) -> str:
    assert img_channels >= 1, f"img_channel must be 1(greyscale) or 3(RGB). got={img_channels}"

### DCGAN Layer ###
class BasicGenerator(nn.Module):
    """ DCGAN Generator layer"""
    __constants__ = ['dim', 'img_channels', 'z_dim']

    def __init__(self, dim: int, img_channels: int, z_dim=100) -> None:
        super(BasicGenerator, self).__init__()
    
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
    
    """def save(self):
        SaveLoadData.save_model(self.models)"""


class BasicDiscriminator(nn.Module):
    __constants__ = ['dim, img_channels, filters']

    def __init__(self, dim: int, img_channels: int, filters=64) -> None:
        super(BasicDiscriminator, self).__init__()

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