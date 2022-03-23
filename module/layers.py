import torch.nn as nn 
from torchvision.models import vgg19

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        vgg19_model = vgg19(pretrained=True)

        self.feature_extractor = nn.Sequential(
            *list(vgg19_model.features.children())[:18]
        )
    
    def forward(self, image):
        return self.feature_extractor(image)


class ResidualBlock(nn.Module):
    def __init__(self, in_features:int , kernel_size:int = 3, stride:int = 1, padding=1):
        super(ResidualBlock, self).__init__()

        self.residual_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size, stride, padding),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),

            nn.Conv2d(in_features, in_features, kernel_size, stride, padding),
            nn.BatchNorm2d(in_features, 0.8),
        )
    
    def forward(self, x):
        return x + self.residual_block(x)