## TODO 데이터셋 다운로드 하는거 따로 빼기 

from matplotlib.transforms import Transform
from torchvision import transforms, datasets 
from torch.utils.data import DataLoader


CELEBA_PATH = f'../data/celebA'
MNIST_PATH = f'../data/mnist'
EMNIST_PATH = f'../data/emnist'
CIFAR10_PATH = f'../data/cifar10'


TRANSFORM = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])

## `torchvision.datasets.CeleA` have some BadZipFile error
def load_celeba_dataset(path: str = CELEBA_PATH) -> any:
    celeba = datasets.CelebA(
        root = path,
        split = 'train',
        download = True, 
        transform = TRANSFORM
    )

    return celeba

def load_cifar_dataset(path: str = CIFAR10_PATH) -> any:
    cifar = datasets.CIFAR10(
        root = path,
        train = True,
        download = True,
        transform = TRANSFORM
    )

    return cifar 


def load_emnist_dataset(path: str = EMNIST_PATH) -> any:
    emnist = datasets.EMNIST(
        root = path,
        train = True,
        download = True,
        transform = TRANSFORM
    )

    return emnist 



def load_mnist_dataset(path: str = MNIST_PATH):
    mnist = datasets.MNIST(
        root = path,
        train = True,
        download = True,
        transform = TRANSFORM
    )

    return mnist


if __name__ == '__main__':
    load_cifar_dataset()