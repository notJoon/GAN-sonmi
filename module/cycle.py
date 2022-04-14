## TODO Cycle-GAN Implementation.
## TODO 프로토타이핑 한 이후에 models.py로 이동. 

from typing import Any
import torch 
import torch.nn as nn 

class Block(nn.Module):
    __constant__ = ['in_channels', 'out_channels', 'stride']

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size = 4, 
                stride = stride, 
                padding = 1, 
                bias = True, 
                padding_mode = 'reflect'
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.conv(x)

class ConvBlock(nn.Module):
    __constants__ = ['in_channels', 'out_channels']

    def __init__(self, in_channels: int, out_channels: int, 
                    down=True, use_activation=True, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode='reflect', **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_activation else nn.Identity()
        )
    
    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    __constants__ = ['channels']

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_activation=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class CycleGenerator(nn.Module):
    __constants__ = ['img_channels', 'num_features', 'num_residual']

    def __init__(self, img_channels: int, num_features=64, num_residual=9) -> None:
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                img_channels,
                out_channels=num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode='reflect'
            ),
            nn.ReLU(inplace=True)
        )

        self.down_blocks = nn.ModuleList([
            ConvBlock(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
            ConvBlock(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1)
        ])

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residual)]
        )

        self.up_block = nn.ModuleList([
            ConvBlock(num_features*4, num_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ConvBlock(num_features*2, num_features, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
        ])

        self.last = nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect')
    
    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)

        x = self.residual_blocks(x)
        for layer in self.up_block:
            x = layer(x)
            
        return torch.tanh(self.last(x))


class CycleDiscriminator(nn.Module):
    def __init__(self, in_channels=3, features =[64, 128, 256, 512]) -> None:
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels=features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode='reflect',
            ),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(
                in_channels, 
                out_channels = feature, 
                stride = 1 if feature == features[-1] else 2
            ))

            in_channels = feature

        layers.append(nn.Conv2d(
            in_channels, 
            out_channels = 1, 
            kernel_size=4, 
            stride=1, 
            padding=1, 
            padding_mode='reflect'
        ))

        self.model = nn.Sequential(*layers)
        #print(self.model)

    def forward(self, x):
        x = self.initial(x)

        # output make sure that between [1, 0]
        return torch.sigmoid(self.model(x))
