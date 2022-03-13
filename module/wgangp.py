import torch
import torch.nn as nn 
import numpy as np

##TODO implementing gradient penalty and other wgan-gp modules 

def get_gradient():
    ... 

def gradient_penalty(gradient) -> torch.Tensor:
    '''
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    
    Parameters:
        * gradient: the gradient of the critic's scores, with respect to the mixed image
    Returns:
        * penalty: the gradient penalty
    '''

    ## flatten the gradients -> each rows captures one image 
    gradient = gradient.view(len(gradient), -1)

    ## calculate L2-norm (Euclidean distance)
    gradient_norm = gradient.norm(2, dim=1)

    penalty = torch.mean((gradient_norm-1) ** 2)
    return penalty 