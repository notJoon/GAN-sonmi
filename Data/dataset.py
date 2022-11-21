import torch 
from torch.utils.data import Dataset, DataLoader

PATH = r'...'

class ImageDataset(Dataset):
    def __init__(self, path: str, transforms=None) -> None:
        self.path = path
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.path)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img = self.path[idx]

        return img 

"""
BATCH = 32
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

dataset = ImageDataset(PATH, transforms=transforms)
dataloader = DataLoader(dataset, batch_size=BATCH, shuffle=True)
"""