import torch
import torch.nn.functional as F
import math
import scipy
import numpy as np
import tqdm 

def sampler(args, config, x, y, model, sde, levy,
                masked_data=None, mask=None, t0=None, device='cuda'):
    
    if args.sample_type not in ['sde', 'ode', 'sde_imputation']:
        raise Exception("Invalid sample type")
    
    is_isotropic= config.diffusion.is_isotropic 
    steps = args.nfe
    eps = 1e-5
    method = args.sample_type
    
    
    def score_model(x, t):
        
        if config.model.is_conditional:
            out = model(x, t, y)
        else:
            out = model(x, t)
        return out



    def sde_score_update(x, s, t):
        """
        input: x_s, s, t
        output: x_t
        """
        score_s = score_model(x, s) * torch.pow(sde.marginal_std(s), -(2-1))[:,None,None,None]
        
        beta_step = sde.beta(s) * (s - t)


        score_s = score_model(x, s) * torch.pow(sde.marginal_std(s) + 1e-5, -(2-1))[:,None,None,None]
        e_B = torch.randn_like(x).to(device)

        x_coeff = 1 + beta_step/2
        score_coeff = beta_step
        noise_coeff = torch.pow(beta_step, 1 / 2)
        x_t = x_coeff[:, None, None, None] * x + score_coeff[:, None, None, None] * score_s + noise_coeff[:, None, None,None] * e_B


       
        return x_t
    


    # Sampling steps    
    timesteps = torch.linspace(sde.T, eps, steps + 1).to(device)  # linear
    
    
    with torch.no_grad():

        for i in tqdm.tqdm(range(steps)):
            vec_s, vec_t = torch.ones((x.shape[0],)).to(device) * timesteps[i], torch.ones((x.shape[0],)).to(device) * timesteps[i + 1]
            x = sde_score_update(x, vec_s, vec_t)
            # clamp threshold : re-normalization
            if config.sampling.clamp_threshold :
                size = x.shape
                l = len(x)
                x = x.reshape((l, -1))
                indices = x.norm(dim=1) > config.sampling.clamp_threshold
                x[indices] = x[indices] / x[indices].norm(dim=1)[:, None] * config.sampling.clamp_threshold
                x = x.reshape(size)

    return x
