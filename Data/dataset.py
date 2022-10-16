from typing import Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

PATH = 'some/path/to/data'

class ImageDataset(Dataset):
    def __init__(self, csv, image_folder, transform) -> None:
        self.csv = csv
        self.transform = transform
        self.image_folder = image_folder
        self.foramt = '*' + '.png'
        self.image_names = self.csv[:]['Id']
        self.labels = np.array(self.csv.drop(['Id', 'Label'], axis=1))

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx) -> Dict:
        image = cv2.imread(self.image_folder + self.image_names.iloc[idx]+'.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        image = self.transform(image)
        targets = self.labels[idx]

        sample = {'image': image, 'labels': targets}

        return sample
        

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

df = pd.read_csv(PATH)
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
BATCH_SIZE = 32

train_dataet = ImageDataset(train_set, img_folder, train_transform)
test_dataset = ImageDataset(test_set, img_folder, test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)

## Get a batch of training data
images = next(iter(test_loader))

## Make a grid from batch
out = torch.cat((images['image'][:4]), dim=0)

imshow(out)