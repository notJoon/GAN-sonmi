from glob import glob
from PIL import Image
import torch 
from torch.utils.data import Dataset, DataLoader
import pickle as pkl 

##TODO make labels

class CustomData(Dataset):
    def __init__(self, path: str, train=True, transform=None):
        self.path = path 

        if train:
            self.img_path = path + '/train'
        else:
            self.img_path = path + '/test'
        
        self.img_list = glob(self.img_path + '/*.png')
        self.transform = transform

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)
        
        return img 
        