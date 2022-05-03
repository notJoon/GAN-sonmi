from torch.utils.data import Dataset
from PIL import Image 
import os 
import numpy as np 

class HorseZebra(Dataset):
    def __init__(self, root_zebra: str, root_horse: str, transform=None) -> None:
        self.root_zebra = root_zebra
        self.root_horse = root_horse
        self.transform = transform 

        self.zebra_imges = os.listdir(root_zebra)
        self.horse_imges = os.listdir(root_horse)
        self.length_dataset = max(len(self.zebra_imges), len(self.horse_imges))
        self.zebra_len = len(self.zebra_imges)
        self.horse_len = len(self.horse_imges)
    
    def __len__(self) -> int:
        return self.length_dataset
    
    def __getitem__(self, index) -> tuple[np.ndarray, np.ndarray]:
        zebra_img = self.zebra_imges[index % self.zebra_len]
        horse_img = self.horse_imges[index % self.horse_len]

        zebra_path = os.path.join(self.root_zebra, zebra_img)
        horse_path = os.path.join(self.root_horse, horse_img)

        zebra_img = np.array(Image.open(zebra_path).convert("RGB"))
        horse_img = np.array(Image.open(horse_path).convert("RGB"))

        if self.transform:
            augmentation = self.transform(image=zebra_img, image0=horse_img)
            zebra_img = augmentation['image']
            horse_img = augmentation['image0']
        
        return zebra_img, horse_img