from glob import glob
from PIL import Image
import torch 
from torch.utils.data import Dataset, DataLoader

##TODO make custom dataset
##TODO 이미지 리스트 만들기 
## 2. augmentation 과정 추가 
## 3. resize/crop 한 후에 preprocessing 파일 지워도 될거 같으면 정리  

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


if __name__ == "__main__":
    dataset = CustomData(10)
    dataloader = DataLoader(
                    dataset=dataset,
                    batch_size=2,
                    shuffle=True,
                    drop_last=False,
                )
    epochs = 10 
    for epoch in range(epochs):
        print(f'epoch: {epoch+1}')
        for batch in dataloader:
            print(batch)