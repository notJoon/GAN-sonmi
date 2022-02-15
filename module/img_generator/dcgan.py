"""TODO 
파이토치로 기본적인 것들은 구현하고 그 이후에 mnist로 모델 돌려봐야 할 듯
적당한 결과가 나오면 그 이후에 자체제작한 이미지 데이터셋으로 돌려보고 보완하든지 
그러고 아 이건 된다 싶을때 모델 배포 하기. 빠르면 3월 중으로

https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py
""" 
import os
import math
from typing import Any
import numpy as np
from tqdm import tqdm 

import torch

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

CUDA = True if torch.cuda.is_available() else False
os.makedirs('images', exist_ok=True)

## Load datasets (mnist)
IMG_SIZE: int = 80
BATCH_SIZE: int = 125

os.makedirs("/Users/not_joon/projects/GAN-sonmi/data/mnist", exist_ok=True)
dataloader = DataLoader(datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(IMG_SIZE), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

### DCGAN Model class
class DCGAN(nn.Module):
    def __init__(self, img_width: int, img_height: int, img_channels: int, 
                latent_dim: int):
        super().__init__(self, DCGAN)
        
        self.img_width = img_width
        self.img_height = img_height
        self.img_channels = img_channels
        self.img_shape = (self.img_width, self.img_height, self.img_channels)

        self.latent_dim = latent_dim
        self.alpha = 0.2             ##for LeakyRelu

    ## discriminator
    def discriminator(self, img):
        """ receive generated imsge. then check is this looks real or not 
                * img : generated image
        """

        self.model_disc = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(self.alpha, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(self.alpha, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        img_flat = img.view(img.size(0), -1)
        validity = self.model_disc(img_flat)

        return validity
            
            

     ## generator
    def generator(self, z):
        """ receive noise and generate image.
        * z : noise 
        """
        self.model_gen = nn.Sequential(
            # Layer 1 : in -> latent_dim, out -> 128
            nn.Linear(self.latent_dim, 128),
            nn.BatchNorm1d(128, 0.8),
            nn.LeakyReLU(self.alpha, inplace=True),

            # layer 2 : im -> 128, out -> 256
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(self.alpha, inplace=True),

            # layer 3 : in -> 256, out -> 512
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(self.alpha, inplace=True),

            # layer 4: 512, out -> 1024 
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(self.alpha, inplace=True),


            # layer 5: in -> 1024, out -> img_shape 
            nn.Linear(1024, self.img_shape),
            nn.Tanh() 

        )

        img = self.model_gen(z)
        img = img.view(img.size(0), self.img_shape)

        return img
            

loss = nn.BCELoss()
disc = DCGAN.discriminator()
gen = DCGAN.generator()

optim_D = torch.optim.Adam(disc.parameters(), lr=0.001, betas=(0.9, 0.28))
optim_G = torch.optim.Adam(gen.parameters(), lr=0.001, betas=(0.9, 0.28))

Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

## Train
EPOCHS = 100

for epoch in range(EPOCHS):
    for i, (imgs, _) in enumerate(dataloader):
    
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        real_imgs = Variable(imgs.type(Tensor))

        # generator

        optim_G.zero_grad()

        # input: sample noise(z)
        latent_dim = 64

        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))

        generated_imgs = gen(z)
        generator_loss = loss(disc(generated_imgs), valid)

        generator_loss.backward()
        optim_G.step()

        # discriminator 
        optim_D.zero_grad()

        real_loss = loss(disc(real_imgs), valid)
        fake_loss = loss(disc(generated_imgs.detach()), valid)

        discriminator_loss = (real_loss + fake_loss) / 2

        discriminator_loss.backward()
        optim_D.step()

        # update loss 
        print(f'DISCRIMINATOR LOSS: {discriminator_loss}, GENERATOR LOSS: {generator_loss}')

        # save img
print('done')
    