from typing import Union
import torch

def gradient_penalty(critic, labels, real, fake, device='cpu', mode='wgan'):
    BATCH_SIZE, CHANNELS, HEIGHT, WIDTH = real.shape 
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, CHANNELS, HEIGHT, WIDTH).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)

    ## calculate critic scores 
    if mode == 'wgan':
        mixed_scores = critic(interpolated_images)
    if mode == 'conditional':
        mixed_scores = critic(interpolated_images, labels)

    gradient = torch.autograd.grad(
        input=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1)**2)

    return gradient_penalty