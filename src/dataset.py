import glob 
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms 
from PIL import Image 
## https://discuss.pytorch.org/t/resizing-dataset/75620/5

class chimchakDataset(DataLoader):
    def __init__(self, path: str, train=True, transform=None) -> None:
        ## 이미지 전처리 과정 
        ## 그레이스케일 변환이나, 사이즈 조절 등 여러가지 하면 될긋
        ## 일단 single0-label로 만든 다음에 추후 multi-label로 변경
        self.path = path 

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

    def __len__(self) -> int:
        ## 데이터셋의 길이 반환
        return len(self.x_data)

    def __getitem__(self, idx: int):
        ## 데이터셋에서 특정한 인덱스를 가진 샘플 가져오기 
        if self.transform is not None:
            ...

