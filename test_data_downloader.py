## TODO 데이터셋 다운로드 하는거 따로 빼기 

from torchvision import transforms, datasets 
from torch.utils.data import DataLoader

import glob 
from sklearn.model_selection import train_test_split

CELEBA_PATH = f'/Users/not_joon/projects/GAN-sonmi/data/celebA'
MNIST_PATH = f'/Users/not_joon/projects/GAN-sonmi/data/mnist'

class CelebALoader():
    def __init__(self, path=CELEBA_PATH):
        self.path = path

    def load_celeba_data(self):
        self.data = datasets.CelebA(
            root = self.path,
            train = True,
            download = True, 
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ])
        )



class MNISTLoader():
    def __init__(self, path=MNIST_PATH):
        self.path = path 

    def load_mnist_data(self):
        self.data = datasets.MNIST(
            root = self.path,
            train = True,
            download = True,
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ])
        )


if __name__ == '__main__':
    CelebALoader(CELEBA_PATH)
    MNISTLoader(MNIST_PATH)