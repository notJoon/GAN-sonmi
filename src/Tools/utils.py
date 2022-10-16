import matplotlib.pyplot as plt
import numpy as np

import torchvision.utils as vutils
import torchvision.transforms as transforms
import torch 

from skimage.exposure import match_histograms

# ref: https://huggingface.co/spaces/therealcyberlord/abstract-art-generation/blob/main/utils.py

def color_histogram_mapping(images, ref) -> torch.tensor:
    matched = []

    for i in range(len(images)):
        _matched = match_histograms(
            images[i].permute(1, 2, 0).numpy(), 
            ref[i].permute(1, 2, 0).numpy, 
            channel_axis = -1,
        )
    
    return torch.tensor(np.array(matched)).permute(0, 3, 1, 2)

def visualize(images, seed=42):
    plt.figure(figsize=(16, 16))
    plt.title(f"Seed: {seed}")
    plt.imshow(np.transpose(vutils.make_grid(images, padding=2, nrow=5, normalize=True), (2, 1, 0)))
    plt.show()
    # plt.axis("off")

def denormalize(images):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    inv_norm = transforms.Normalize(
        mean = [-m/s for m, s in zip(mean, std)],
        std = [1/s for s in std]
    )

    return inv_norm(images)