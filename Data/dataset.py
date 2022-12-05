import torch 
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

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

BATCH = 32
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# dataset = MalnyeonDataset(load_data)
# dataloader = DataLoader(
#     dataset=dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
# )

# for data in dataloader:
#     img = data
#     plt.imshow(
#         torchvision.utils.make_grid(img, normalize=True).permute(1, 2, 0),
#         interpolation="bicubic",
#     )
#     plt.show()

dataset = ImageDataset(PATH, transforms=transforms)
dataloader = DataLoader(dataset, batch_size=BATCH, shuffle=True)
