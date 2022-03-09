## TODO
# Finish the SRGAN 

import numpy as np
import os
import glob, random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid
from layers import FeatureExtractor, ResidualBlock

import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import train_test_split

random.seed(42)

import warnings
warnings.filterwarnings("ignore")

EPOCHS = 2
BATCH_SIZE = 16
CHANNELS = 3

load_pretrained_models = True
PATH = f'../GAN-sonmi/data/clebA'

## for adam
lr = 0.00008
b1 = 0.5
b2 = 0.999

# epoch from which to start lr decay
decay_epoch = 100

# number of cpu threads to use during batch generation
#n_cpu = 8

# high res. image height and width
hr_height, hr_width = 256, 256


os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

cuda = torch.cuda.is_available()
hr_shape = (hr_height, hr_width)


# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

class ImageDataset(Dataset):
    def __init__(self, files, hr_shape):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // 4, hr_width // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.files = files
    
    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)

train_paths, test_paths = train_test_split(
        sorted(glob.glob(PATH + "/*.*")), 
        test_size = 0.02, 
        random_state = 42
    )

train_dataloader = DataLoader(
        ImageDataset(train_paths, hr_shape=hr_shape), 
        batch_size = BATCH_SIZE, 
        shuffle = True, 
        #num_workers = n_cpu
    )

test_dataloader = DataLoader(
        ImageDataset(test_paths, hr_shape=hr_shape), 
        batch_size = int(BATCH_SIZE * 0.75), 
        shuffle = True, 
        #num_workers = n_cpu
    )


class Discriminator(nn.Module):
    def __init__(self, input_shape: int):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape

        in_channels, in_height, in_width = self.input_shape
        patch_height, patch_width = int(in_height / 2 ** 4), int(in_width / 2 ** 4)

        self.out_shape = (1, patch_height, patch_width)

        def discriminator_block(in_filters: int, out_filters: int, first_block=False) -> list:
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
