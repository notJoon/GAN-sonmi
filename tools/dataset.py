import albumentations as A 
from albumentations.pytorch import ToTensorV2
from matplotlib.colors import Normalize
from pandas.core.common import flatten
import matplotlib.pyplot as plt 
import cv2, glob, numpy
import random, torch 

train_data_path = f'images/train'
test_data_path = f'images/test'

train_image_paths = []
labels = []

"""define transform"""
train_transform = A.Compose(
    A.smallest_max_size(max_size=350),
    A.ShiftScaleRotate(shift_limit=0.07, scale_limit=0.05, rotate_limit=30, p=0.5),
    A.MultiplicativeNoise(),
    A.Normalize(),
    ToTensorV2()
)

test_transform = A.Compose(
    A.smallest_max_size(max_size=350),
    A.CenterCrop(height=256, width=256),
    A.Normalize(),
    ToTensorV2()
)


""" Create train, test, validation set """
for data_path in glob.glob(train_data_path + '/*'):
    labels.append(data_path.split('/')[-1])
    train_image_paths.append(glob.glob(data_path) + '/*')

train_image_paths = [flatten(train_image_paths)]
random.shuffle(train_image_paths)

#print('train_image_path example: ', train_image_paths[0])
#print('class example: ', labels[0])

"""dataset class"""
class WebToonFaceDataset:
    def __init__(self, path: str, transform=None) -> None:
        self.path = path 

    def __len__(self) -> int: return len(self.path)

    def __getitem__(self, idx: int):
        file_path = self.path[idx]
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        label = file_path.split('/')[-2]
        ... 