from typing import List

import torch
import torch.nn as nn

### DCGAN Layer ###
class BasicGenerator(nn.Module):
    __constants__ = ['dim', 'img_channels', 'z_dim']

    def __init__(self, dim: int) -> None:
        self.channels = 1
        self.dim = dim
        self.z_dim = 100 
    
    def build_layers(
        self, in_filters:int, out_filters:int, first_block:bool
        ) -> List[nn.Module]:

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
    
    def build_model(self) -> List[nn.Module]:
        layers = []
        in_filters = self.z_dim

        for i, out_filters in enumerate([self.dim*8, self.dim*4, self.dim*2, self.dim]):
            layers.extend(self.build_layers(
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
        return nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        return self.build_model(x)
    
    #TODO make save and load model parameter method


class BasicDiscriminator(nn.Module):
    __constants__ = ['dim, img_channels, filters']

    def __init__(self, dim: int) -> None:
        super(BasicDiscriminator, self).__init__()
        self.channels = 1
        self.filters = 64
        self.dim = dim

    def build_block(self, in_filters: int, out_filters: int) -> List[nn.Module]:
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

    def build_model(self) -> List[nn.Module]:
        layers = []
        layers.append(nn.Conv2d(
            in_channels = self.channels, 
            out_channels = self.filters,
            kernel_size = 4,
            stride = 2,
            padding = 1))
        layers.append(nn.Sigmoid())
        
        input = self.filters
        for _, output in enumerate([self.dim*2, self.dim*4, self.dim*8]):
            layers.extend(self.build_block(
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

        return nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        models = self.build_model(x)
        return models.squeeze()  


#TODO build conditional GAN 

if __name__ == '__main__':
    disc = BasicDiscriminator(32)
    model = disc.build_model()
    print(model)