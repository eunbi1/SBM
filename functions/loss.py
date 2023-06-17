import torch
import torch.nn.functional as F
import math

def loss_fn(model, sde,
            x0,
            t,
            e,
            config,
            y=None):

    sigma = sde.marginal_std(t)
    x_coeff = sde.diffusion_coeff(t)
    
    x_t = x0 * x_coeff[:,None,None,None] + e * sigma[:,None,None,None]


    score = - e


    output = model(x_t, t, y)
    
    d = x0.shape[-1] * x0.shape[-2] * x0.shape[-3]
    
    out = F.smooth_l1_loss(output, score, beta=1, size_average=True, reduction='mean') * d

    return out
