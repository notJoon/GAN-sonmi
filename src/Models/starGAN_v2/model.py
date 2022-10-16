from typing import Any, List

import math
import numpy as np

import torch
from torch import Tensor, nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(
        self, dim_in: int, dim_out: int, activation=nn.ReLU(0.2),
        normalize=False, downsample=False, **kwargs) -> None:
        super().__init__()
        self.activation = activation 
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = (dim_in != dim_out)
        self._build_weights(dim_in, dim_out)
    
    def _build_weights(self, dim_in: int, dim_out: int) -> None:
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in,dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
        
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _skip(self, x: Tensor) -> Tensor:
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        
        return x 
    
    def _residual(self, x: Tensor) -> Tensor:
        if self.normalize:
            x = self.norm1(x)
        
        x = self.activation(x)
        x = self.conv1(x)

        if self.downsample(x):
            x = F.avg_pool2d(x, 2)
        
        if self.normalize:
            x = self.norm2(x)
        
        x = self.activation(x)
        x = self.conv2(x)

        return x 
    
    def forward(self, x: Tensor) -> float:
        x = self._skip(x) + self._residual(x)
        return x / math.sqrt(2)

class AdaInstanceNorm(nn.Module):
    def __init__(self, style_dim: int, num_features: int) -> None:
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)
    
    def forward(self, x, s) -> float:
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta 

class AdaResBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, style_dim=64, w_hpf=0,
    activation=nn.LeakyReLU(0.2), upsample=False) -> None:
        super().__init__()

        self.w_hpf = w_hpf
        self.activation = activation
        self.upsample = upsample
        self.learned_sc = (dim_in != dim_out)
        self._build_weights(dim_in, dim_out, style_dim)
    
    def _build_weights(self, dim_in, dim_out, style_dim) -> None:
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)

        self.norm1 = AdaInstanceNorm(style_dim, dim_in)
        self.norm2 = AdaInstanceNorm(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
    
    def _skip(self, x) -> Any:
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x 
    
    def _residual(self, x, s) -> Any:
        x = self.norm1(x, s)
        x = self.activation(x)

        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        
        x = self.norm1(x, s)
        x = self.norm2(x, s)
        x = self.activation(x)
        x = self.conv2(x)

        return x 

    def print_model(self, model, name) -> None:
        params = 0
        for param in model.parameters():
            params += param.numel()
        
        print(model)
        print(name)
        print(f'number of parameters: {params}')

    def forward(self, x, s):
        y = self._residual(x, s)
        if self.w_hpf == 0:
            y = (y + self._skip(x)) / math.sqrt(2)

class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=16, domains=2) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.domains = domains
        self.shared = self._build_shared()
        self.unshared = self._build_unshared()

    def _build_shared(self, features=512, iter=3) -> List[nn.Module]:
        layers = []
        layers.append(nn.Linear(self.latent_dim, features))
        layers.append(nn.ReLU())

        for _ in range(iter):
            layers.append(nn.Linear(features, features))
            layers.append(nn.ReLU())
        
        return nn.Sequential(*layers)
    
    def _build_unshared(self, features=512) -> List[nn.Module]:
        layers = []
        layers.append(nn.ModuleList())

        for _ in range(self.domains):
            layers.append(nn.Linear(features, features))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(features, self.style_dim))

        return nn.Sequential(*layers)
    
    def forward(self, z, y) -> Any:
        h = self.shared(z)
        res = []
        for layer in self.unshared:
            res += [layer(h)]

        # (batch, domains, style_dim)
        res = torch.stack(res, dim=1)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)

        # (batch, style_dim)
        s = res[idx, y]
        return s 

class Generator(nn.Module):
    def __init__(self, img_size=256, dim=64, max_conv_dim=512) -> None:
        super().__init__()
        dim_in = 2**14 // img_size
        self.img_size = img_size
        self.dim = dim 
        self.max_conv_dim = max_conv_dim
    
    def _build_blocks(self):
        nums = int(np.log2(self.img_size)) - 4
        #TODO

    def print_network(self, model, name):
        num_params = 0
        for param in model.parameters():
            num_params += param.numel()
        
        print(model)
        print(name)
        print(f'total parameters: {num_params}')

class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, domains=2, max_conv_dim=512) -> None:
        super().__init__()
        self.img_size = img_size

class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512) -> None:
        super().__init__()
        self.img_size = img_size
        self.num_domains = num_domains
        self.max_conv_dim = 512
    
    def _build_blocks(self, activ=nn.LeakyReLU(0.2)) -> List[nn.Module]:
        dim_in = 2**14 // self.img_size
        blocks = []
        blocks.append(nn.Conv2d(3, dim_in, 3, 1, 1))
        
        nums = int(np.log2(self.img_size)) - 2
        for _ in range(nums):
            dim_out = min(dim_in * 2, self.max_conv_dim)
            blocks.append(ResBlock(dim_in, dim_out, downsample=True))
            dim_in = dim_out 
        
        blocks.append(activ)
        blocks.append(nn.Conv2d(dim_out, dim_out, 4, 1, 0))
        blocks.append(activ)
        blocks.append(nn.Conv2d(dim_out, self.num_domains, 1, 1, 0))

        self.model = nn.Sequential(*blocks)
        return self.model

    def print_network(self, model, name):
        num_params = 0
        for param in model.parameters():
            num_params += param.numel()

        print(model)
        print(name)
        print(f'number of params: {num_params}')

    def forward(self, x: Tensor, y: Tensor) -> any:
        res = self.model(x)
        res = res.view(res.size(0), -1)
        i = torch.LongTensor(range(y.size(0))).to(y.device)
        res = res[i, y]
        return res 

if __name__ == '__main__':
    base = MappingNetwork()
    build = base._build_unshared()
    print(build)