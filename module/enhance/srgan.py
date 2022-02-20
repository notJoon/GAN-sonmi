## TODO
# Finish the SRGAN 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        vgg19_model = vgg19(pretrained=True)

        self.feature_extractor = nn.Sequential(
            *list(vgg19_model.features.children())[:18]
        )


class ResidualBlock(nn.Module):
    def __init__(self, in_features:int , kernel_size:int = 3, stride:int = 1, padding:int = 1):
        super(ResidualBlock, self).__init__()

        self.residual_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size, stride, padding),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),

            nn.Conv2d(in_features, in_features, kernel_size, stride, padding),
            nn.BatchNorm2d(in_features, 0.8),
        )
    
    def forward(self, x):
        return x + self.residual_block(x)

class Discriminator(nn.Module):
    def __init__(self, input_shape: int):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape

        in_channels, in_height, in_width = self.input_shape
        patch_height, patch_width = int(in_height / 2 ** 4), int(in_width / 2 ** 4)

        self.out_shape = (1, patch_height, patch_width)

        def discriminator_block(in_filters: int, out_filters: int, first_block: bool = False) -> list:
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, pading=1))

            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers
        
        layers = []
        in_filters = in_channels

        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, firtst_block=(i == 0)))

            in_filters = out_filters
        
        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

        
class GeneratorResNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, n_resnet_blocks: int = 16):
        super(GeneratorResNet, self).__init__()
        
        ### Layer 1 ###
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        ## ResNet Block ## 
        res_blocks = []

        for _ in range(n_resnet_blocks):
            res_blocks.append(ResidualBlock(64))
        
        self.res_blocks = nn.Sequential(*res_blocks)

        ### Layer 2, post residual blocks ###
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8)
        )

        ## Upsampling
        upsampling = []

        for out_features in range(2):
            upsampling += [
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU()
            ]
        
        self.upsampling = nn.Sequential(*upsampling)

        ### output
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)

        out2 = self.conv2(out)
        out = torch.add(out1, out2)

        out = self.upsampling(out)

        out = self.conv3(out)

        return out 
