## https://github.com/yellowjs0304/3-min-pytorch_study/blob/master/09-%EA%B2%BD%EC%9F%81%ED%95%98%EB%A9%B0_%ED%95%99%EC%8A%B5%ED%95%98%EB%8A%94_GAN/gan.ipynb

## TODO 터미널에 ansi 적용
## FIXME 
# 1. discriminator 부분에서 오버피팅 되는거 같음
# 2. loss 계산 안 됨  

import datetime
import os 
import torch
import torch.nn as nn
import torch.optim as optim 
import matplotlib.pyplot as plt 
import numpy as np

## for test modules we will using MNIST dataset
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader

BATCH_SIZE = 100
EPOCHS = 1
PATH = '/Users/not_joon/projects/GAN-sonmi/data/mnist'

## for file name 
utc_datetime = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H%MZ")
filename = '/Users/not_joon/projects/GAN-sonmi/saved_imgs/fig_%s.png' % utc_datetime

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("start downloading \n")

## Load dataset for test
data = datasets.MNIST(
    root = PATH,
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])
)

print("Data has Loaded \n")

data_loader = DataLoader(
    dataset = data,
    batch_size = BATCH_SIZE,
    shuffle = True
)

Discriminator = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

Generator = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 784),
            nn.Tanh()
        )
    
disc = Discriminator.to(DEVICE)
gen = Generator.to(DEVICE)
loss = nn.BCELoss()

disc_optim = optim.Adam(disc.parameters(), lr=0.0002)
gen_optim = optim.Adam(gen.parameters(), lr=0.002)

total_step = len(data_loader)

## Train
print('START TRAIN')
for epoch in range(EPOCHS):
    for i, (images, _) in enumerate(data_loader):
        images = images.reshape(BATCH_SIZE, -1).to(DEVICE)

        ## generate real, fake labels
        real = torch.ones(BATCH_SIZE, 1).to(DEVICE)
        fake = torch.zeros(BATCH_SIZE, 1).to(DEVICE)

        ## discriminator: calculate loss 
        outputs = disc(images)
        disc_real_loss = loss(outputs, real)
        real_score = outputs

        ## generate fake images
        z = torch.randn(BATCH_SIZE, 64).to(DEVICE)
        fake_images = gen(z)

        ## calculate loss between real and fake images 
        ## is it looks like real image or not? 
        outputs = disc(fake_images)
        disc_fake_loss = loss(outputs, fake)
        fake_score = outputs

        ## calculate discriminator's loss 
        disc_loss = disc_real_loss + disc_fake_loss

        """ train discriminator """
        disc_optim.zero_grad()
        gen_optim.zero_grad()

        disc_loss.backward()
        disc_optim.step()

        ## calculate generator's loss 
        fake_images = gen(z)
        outputs = disc(fake_images)
        gen_loss = loss(outputs, real)

        """ train generator """
        disc_optim.zero_grad()
        gen_optim.zero_grad()

        gen_loss.backward()
        gen_optim.step()

    print(f'EPOCH: {epoch + 1}/{EPOCHS}, disc_loss: {disc_loss.item():.3f}, gen_loss: {gen_loss.item():.3f}, SCORE: {real_score.mean().item():.3f}')


z = torch.randn(BATCH_SIZE, 64).to(DEVICE)
fake_images = gen(z)

for i in range(10):
    image = np.reshape(fake_images.data.cpu().numpy()[i], (28, 28))
    plt.imshow(image, cmap='gray')
    plt.savefig(filename)
    plt.show()

print('FINISHED')